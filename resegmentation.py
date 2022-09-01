import os
import pickle
import queue
import threading
from concurrent.futures import Executor
from typing import Callable, List, Mapping, Optional

import loky
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
from morphocut.annotation import DrawContours
from morphocut.contrib.ecotaxa import EcotaxaReader, EcoTaxaReader2, EcotaxaWriter
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
    ImageProperties,
    ImageWriter,
    RegionProperties,
    RGB2Gray,
    filter_objects_by_size,
)
from morphocut.pandas import PandasWriter
from morphocut.str import Format
from morphocut.stream import Filter, Progress, Slice, Unpack
from sklearn.ensemble import RandomForestClassifier


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


def keep_largest_object(mask):
    labels, n_labels = skimage.measure.label(mask, return_num=True)

    areas = [np.sum(labels == l) for l in range(n_labels + 1)]
    areas[0] = 0

    return labels == np.argmax(areas)


def label_ex(mask, merge_radius: Optional[int] = None):
    if merge_radius is None:
        return skimage.morphology.label(mask)

    # Enlarge objects by merge_radius
    label_mask = skimage.morphology.binary_dilation(
        mask, skimage.morphology.disk(merge_radius)
    )

    # Restrict labeled segments to previous (undilated) mask
    return skimage.morphology.label(label_mask) * mask


def _segment(
    image,
    min_size: Optional[int] = None,
    closing_radius=None,
    merge_radius=None,
):
    # Calculate features
    features = skimage.feature.multiscale_basic_features(
        image,
        intensity=True,
        edges=True,
        texture=True,
        sigma_max=32,
        num_workers=1,
    )

    classifier: RandomForestClassifier = get_worker_env("classifier")
    classifier.n_jobs = 1
    classifier.verbose = 0

    # Predict segmentation
    mask = skimage.future.predict_segmenter(features, classifier) == 1

    # TODO: Morphological opening (to remove noisy pixels)

    # Morphological closing (to remove small cracks)
    if closing_radius is not None:
        mask = skimage.morphology.binary_closing(
            mask, skimage.morphology.disk(closing_radius)
        )

    # Remove small objects
    if min_size is not None:
        mask = skimage.morphology.remove_small_objects(
            mask, min_size=min_size, in_place=True
        )

    # Label connected components, merging components closer than merge_radius*2
    return label_ex(mask, merge_radius)


with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

initialize_worker({"classifier": classifier})


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
                        (obj, self.executor.submit(self.clbl, *args, **kwargs))
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


def _calc_metadata(region: RegionProperties, et_obj) -> Mapping:
    """
    Updates posx, posy, and, accordingly, object_id.
    """

    object_id_old = et_obj.object_id
    result: Optional[parse.Result] = objid_parser.parse(object_id_old)

    if result is None:
        raise ValueError(f"Could not parse {object_id_old!r}")

    date, time, msec, seq, posx, posy = result

    y, x, *_ = region.bbox

    posx += x
    posy += y
    seq += region.label - 1

    object_id = objid_pattern.format(date, time, msec, seq, posx, posy)

    meta = {
        **et_obj.meta,
        "object_id": object_id,
        "object_id_old": object_id_old,
        "img_file_name": object_id + os.path.splitext(et_obj.meta["img_file_name"])[1],
        "posx": posx,
        "posy": posy,
    }

    return meta


with Pipeline() as p:
    # archive_fn = Glob("input/*.zip", prefetch=True)
    # archive_fn = "input/export_362_20220215_0946.zip"
    archive_fn = Unpack(
        [
            # "input/export_4871_20220215_0946.zip",
            # "input/export_4621_20220214_1630.zip",
            "input/export_4821_20220214_1630.zip"
        ]
    )
    archive_basename = Call(lambda archive_fn: os.path.basename(archive_fn), archive_fn)

    target_archive_fn = Call(os.path.join, "resegmented", archive_basename)
    seg_archive_fn = Call(os.path.join, "resegmented", "seg-" + archive_basename)

    Progress(Format("Archive {}", archive_basename), unit_scale=True)

    et_obj = EcoTaxaReader2(archive_fn)

    ## Debug: Process only 100 objects
    # Slice(100)

    # Debug()

    # Progress(Format("Load {}", et_obj.object_id), unit_scale=True)

    # valid_buffer = Call(lambda et_obj: et_obj.image_data.getbuffer(), et_obj)
    # Filter(valid_buffer)

    # Load image (grayscale uint8)
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    image = et_obj.get_image(mode="L")

    image_area = Call(lambda image: np.prod(image.shape[:2]), image)

    # Skip small images
    Filter(image_area > (3 * 175) ** 2)

    labels = ParallelCall(
        _segment,
        loky.get_reusable_executor(
            max_workers=6,
            initializer=initialize_worker,
            initargs=({"classifier": classifier}),
            timeout=20,
        ),
    )(
        image,
        # 1000 seems plausible
        min_size=1000,
        # 1 produces best F1 (although larger would further increase recall)
        closing_radius=1,
        # Merge segments closer than 180px to each other to avoid over-segmentation
        merge_radius=90,
    )

    segment_image = Call(
        lambda labels, image: skimage.util.img_as_ubyte(
            skimage.color.label2rgb(labels, image, bg_label=0, bg_color=None)
        ),
        labels,
        image,
    )

    EcotaxaWriter(
        seg_archive_fn,
        [
            # Input image
            ("img/" + et_obj.meta["img_file_name"], image),
            # Image overlayed with labeled segments
            ("overlay/" + et_obj.meta["img_file_name"], segment_image),
            # Mask (1=foreground)
            (
                "mask/" + et_obj.meta["img_file_name"],
                Call(lambda labels: (labels > 0).astype("uint8"), labels),
            ),
        ],
    )

    Progress(et_obj.object_id)

    # region = FindRegions(labels, image, padding=175)

    # resized_image = ExtractROI(image, region)

    # area_fraction = Call(
    #     lambda image, resized_image: np.prod(resized_image.shape[:2])
    #     / np.prod(image.shape[:2]),
    #     image,
    #     resized_image,
    # )

    # # Replace only if actually smaller or multiple objects
    # Filter(
    #     Call(
    #         lambda area_fraction, region: area_fraction < 1.0 or region.max_label > 1,
    #         area_fraction,
    #         region,
    #     )
    # )

    # # Re-calculate metadata (object_id, posx, posy)
    # meta = Call(_calc_metadata, region, et_obj)

    # # Re-calculate features
    # features = CalculateZooProcessFeatures(region, prefix="mc_")

    # EcotaxaWriter(
    #     target_archive_fn,
    #     (meta["img_file_name"], resized_image),
    #     meta=meta,
    #     object_meta=features,
    # )


p.run()
