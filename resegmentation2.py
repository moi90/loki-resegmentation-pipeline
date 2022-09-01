import gzip
import os
import pickle
import queue
import threading
from concurrent.futures import Executor
from typing import Callable, List, Mapping, Optional

import loky
from morphocut.utils import StreamEstimator, stream_groupby
import numpy as np
import parse
import scipy.ndimage as ndi
import skimage.color
import skimage.feature
import skimage.future
import skimage.measure
import skimage.morphology
import skimage.util
from grpc import Future
from morphocut import Pipeline
from morphocut.contrib.ecotaxa import EcotaxaReader, EcotaxaWriter
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.core import (
    Call,
    Node,
    Output,
    ReturnOutputs,
    Stream,
    StreamObject,
    Variable,
    closing_if_closable,
)
from morphocut.file import Glob
from morphocut.image import (
    ExtractROI,
    FindRegions,
    RegionProperties,
)
from morphocut.str import Format
from morphocut.stream import Progress, Unpack
import pandas as pd


class Debug(Node):
    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            for obj in stream:
                print("n_remaining_hint", obj.n_remaining_hint)

                yield obj


def initialize_worker(env):
    global _worker_env
    _worker_env = env


def get_worker_env(key):
    global _worker_env

    return _worker_env[key]


import segmenter


def _predict_scores(
    image,
):
    segmenter_: segmenter.Segmenter = get_worker_env("segmenter")

    features = segmenter_.extract_features(image)
    mask = segmenter_.preselect(image)
    scores = segmenter_.predict_pixels(features, mask)

    return scores


with gzip.open("segmenter.pkl.gz", "rb") as f:
    segmenter_: segmenter.Segmenter = pickle.load(f)
    segmenter_.configure(n_jobs=1, verbose=0)

print(segmenter_)

initialize_worker({"segmenter": segmenter_})


def getname(obj):
    try:
        return obj.__name__
    except AttributeError:
        pass

    return type(obj).__name__


class ParallelCall(Node):
    """
    Call a function with the supplied parameters.

    For every object in the stream, apply ``clbl`` to the corresponding stream variables.

    Args:
        clbl: A callable.
        *args: Positional arguments to ``clbl``.
        **kwargs: Keyword-arguments to ``clbl``.

    Returns:
        Variable: The result of the function invocation.

    Example:
        .. code-block:: python

            def foo(bar):
                return bar

            baz = ... # baz is a stream variable.
            result = PipelinedCall(foo)(baz)
    """

    def __init__(
        self,
        clbl: Callable,
        executor: Executor,
        queue_size=16,
    ):
        super().__init__()

        self.clbl = clbl
        self.executor = executor
        self.queue_size = queue_size

        self.args = None
        self.kwargs = None

        self._output = Variable("output", self)

    def __call__(self, *args, **kwargs) -> Variable:
        self.args = args
        self.kwargs = kwargs
        return self._output

    def transform_stream(self, stream: Stream) -> Stream:
        result_queue = queue.Queue(self.queue_size)

        def submitter():
            with closing_if_closable(stream):
                for obj in stream:
                    args, kwargs = self.prepare_input(obj, ("args", "kwargs"))

                    result_queue.put(
                        (obj, self.executor.submit(self.clbl, *args, **kwargs))  # type: ignore
                    )

                result_queue.put(None)

        t = threading.Thread(target=submitter)
        t.start()

        while True:
            wp = result_queue.get()

            if wp is None:
                break

            obj, future = wp
            obj: StreamObject
            future: Future

            result = future.result()

            obj[self._output] = result

            yield obj

    def __str__(self):

        args = [getname(self.clbl)]
        if self.args is not None:
            args.extend(str(a) for a in self.args)
        if self.kwargs is not None:
            args.extend("{}={}".format(k, v) for k, v in self.kwargs.items())
        return "{}({})".format(self.__class__.__name__, ", ".join(args))


objid_pattern = "{:08d} {:06d}  {:03d}  {:06d} {:04d} {:04d}"
objid_parser = parse.compile(objid_pattern)


def sort_index(df: pd.DataFrame):
    return df.sort_values("object_id")


def _image_weights(shape, eps=0.01):
    weights = np.ones(shape[:2], dtype=float)
    weights[weights.shape[0] // 2, weights.shape[1] // 2] = 0
    ndi.distance_transform_edt(weights, distances=weights)
    return 1 - (weights / (weights.max())) + eps


@ReturnOutputs
@Output("image")
@Output("scores")
@Output("offset")
class Stitch(Node):
    def __init__(self, image, scores, *, groupby, x, y, skip_singletons) -> None:
        super().__init__()

        self.image = image
        self.scores = scores
        self.groupby = groupby
        self.x = x
        self.y = y
        self.skip_singletons = skip_singletons

    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            stream_estimator = StreamEstimator()

            for key, group in stream_groupby(stream, self.groupby):
                group = list(group)

                with stream_estimator.consume(
                    group[0].n_remaining_hint, est_n_emit=1, n_consumed=len(group)
                ) as incoming_group:

                    if len(group) == 1:
                        if not self.skip_singletons:
                            obj = group[0]
                            incoming_group.emit()
                            image, scores, x, y = self.prepare_input(
                                obj, ("image", "scores", "x", "y")
                            )
                            yield self.prepare_output(
                                obj,
                                image,
                                scores,
                                (x, y),
                                n_remaining_hint=incoming_group.emit(),
                            )
                        continue

                    data = pd.DataFrame(
                        [
                            self.prepare_input(obj, ("image", "scores", "x", "y"))
                            for obj in group
                        ],
                        columns=["image", "scores", "x", "y"],
                    )

                    offsx = data["x"].min()
                    data["x"] -= offsx
                    offsy = data["y"].min()
                    data["y"] -= offsy

                    data[["h", "w"]] = data["image"].apply(
                        lambda img: pd.Series(img.shape[:2])
                    )

                    data["xmax"] = data["x"] + data["w"]
                    data["ymax"] = data["y"] + data["h"]

                    first_image = data["image"].iloc[0]
                    first_score = data["scores"].iloc[0]

                    target_shape = (data["ymax"].max(), data["xmax"].max())

                    # Use float accumulator to avoid overflows
                    large_image = np.zeros(
                        target_shape + first_image.shape[2:], dtype=float
                    )
                    large_image_div = np.zeros(target_shape, dtype=float)
                    large_scores = np.zeros(
                        target_shape + first_score.shape[2:], dtype=first_score.dtype
                    )

                    for row in data.itertuples():
                        sl = (slice(row.y, row.y + row.h), slice(row.x, row.x + row.w))
                        weights = _image_weights(row.image.shape)
                        large_image[sl] += weights * row.image
                        large_image_div[sl] += weights

                        np.maximum(large_scores[sl], row.scores, out=large_scores[sl])

                    large_image_div[large_image_div == 0] = 1

                    large_image /= large_image_div
                    large_image = large_image.astype(first_image.dtype)

                    yield self.prepare_output(
                        group[0].copy(),
                        large_image,
                        large_scores,
                        (offsx, offsy),
                        n_remaining_hint=incoming_group.emit(),
                    )


def _calc_metadata(region: RegionProperties, et_obj, offsets) -> Mapping:
    """
    Updates posx, posy, and, accordingly, object_id.
    """

    object_id_old = et_obj.object_id
    result: Optional[parse.Result] = objid_parser.parse(object_id_old)  # type: ignore

    if result is None:
        raise ValueError(f"Could not parse {object_id_old!r}")

    date, time, msec, seq, posx, posy = result

    y, x, *_ = region.bbox

    # Apply offset
    posx = offsets[0] + x
    posy = offsets[1] + y
    seq += region.label - 1

    object_id = objid_pattern.format(date, time, msec, seq, posx, posy)

    meta = {
        # The correct object metadata has to be matched afterwards
        **{k: v for k, v in et_obj.meta.items() if not k.startswith("object_")},
        "object_id": object_id,
        "img_file_name": object_id + os.path.splitext(et_obj.meta["img_file_name"])[1],
        "object_posx": posx,
        "object_posy": posy,
    }

    return meta


def execute(
    archive_fn,
    output_dir="resegmented",
    skip_singletons=False,
    store_seg=False,
    max_workers=6,
):
    with Pipeline() as p:
        archive_fn = Glob(archive_fn, prefetch=True)
        archive_basename = Call(
            lambda archive_fn: os.path.basename(archive_fn), archive_fn
        )

        Progress(archive_basename)

        target_archive_fn = Call(os.path.join, output_dir, archive_basename)
        seg_archive_fn = Call(os.path.join, output_dir, "seg-" + archive_basename)

        Progress(Format("Archive {}", archive_basename), unit_scale=True)

        et_obj = EcotaxaReader(archive_fn, prepare_data=sort_index)

        # Load image (grayscale uint8)
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        image = et_obj.get_image(mode="L")

        scores = ParallelCall(
            _predict_scores,
            loky.get_reusable_executor(
                max_workers=max_workers,
                initializer=initialize_worker,
                initargs=({"segmenter": segmenter_}),
                timeout=20,
            ),
        )(image)

        object_id_fields = Call(objid_parser.parse, et_obj.object_id)

        # Stitch images
        image_large, scores_large, offsets = Stitch(
            image,
            scores,
            groupby=object_id_fields[:3],
            x=object_id_fields[4],
            y=object_id_fields[5],
            skip_singletons=skip_singletons,
        )

        # Post-process classification scores
        labels_large = Call(segmenter_.postprocess, scores_large, image_large)

        if store_seg:
            segment_image = Call(
                lambda labels, image: skimage.util.img_as_ubyte(
                    skimage.color.label2rgb(labels, image, bg_label=0, bg_color=None)
                ),
                labels_large,
                image_large,
            )

            score_image = Call(
                skimage.util.img_as_ubyte,
                scores_large,
            )

            #  Store stitched result
            EcotaxaWriter(
                seg_archive_fn,
                [
                    # Input image
                    ("img/" + et_obj.meta["img_file_name"], image_large),
                    # Image overlayed with labeled segments
                    ("overlay/" + et_obj.meta["img_file_name"], segment_image),
                    ("score/" + et_obj.meta["img_file_name"], score_image),
                    # Mask (1=foreground)
                    (
                        "mask/" + et_obj.meta["img_file_name"],
                        Call(lambda labels: (labels > 0).astype("uint8"), labels_large),
                    ),
                ],
            )

        # Extract individual objects
        region = FindRegions(labels_large, image_large, padding=75)

        obj_image = ExtractROI(image_large, region)

        # Re-calculate metadata (object_id, posx, posy)
        meta = Call(_calc_metadata, region, et_obj, offsets)

        # Re-calculate features
        features = CalculateZooProcessFeatures(region, prefix="mc_")

        EcotaxaWriter(
            target_archive_fn,
            (meta["img_file_name"], obj_image),
            meta=meta,
            object_meta=features,
        )

        Progress(et_obj.object_id)

    p.run()


if __name__ == "__main__":
    import fire

    fire.Fire(execute)
