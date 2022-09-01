from morphocut import Pipeline
from morphocut.contrib.ecotaxa import EcoTaxaReader2, EcotaxaReader, EcotaxaWriter
from morphocut.core import Call, Node, Stream, closing_if_closable
from morphocut.file import Glob
from morphocut.image import RGB2Gray, ImageWriter, ImageProperties
import numpy as np
import os
from morphocut.str import Format
from morphocut.stream import Progress, Filter, Slice
from morphocut.contrib.zooprocess import CalculateZooProcessFeatures
from morphocut.pandas import PandasWriter

os.makedirs("features", exist_ok=True)


class Debug(Node):
    def transform_stream(self, stream: Stream) -> Stream:
        with closing_if_closable(stream):
            for obj in stream:
                print("n_remaining_hint", obj.n_remaining_hint)

                yield obj


with Pipeline() as p:
    archive_fn = Glob("input/*.zip", prefetch=True)
    # archive_fn = "input/export_362_20220215_0946.zip"
    archive_name = Call(
        lambda archive_fn: os.path.splitext(os.path.basename(archive_fn))[0], archive_fn
    )

    features_fn = Format("features/{}.zip", archive_name)

    # # Skip archives that were already processed
    # not_present = Call(lambda features_fn: not os.path.isfile(features_fn), features_fn)
    # Filter(not_present)

    # mask_path = Call(
    #     lambda archive_name: os.path.join("masks", archive_name), archive_name
    # )
    # Call(os.makedirs, mask_path, exist_ok=True)

    Progress(Format("Archive {}", archive_name), unit_scale=True)

    et_obj = EcoTaxaReader2(archive_fn)

    ## Debug: Process only 100 objects
    # Slice(100)

    # Debug()

    # Progress(Format("Load {}", et_obj.object_id), unit_scale=True)

    # valid_buffer = Call(lambda et_obj: et_obj.image_data.getbuffer(), et_obj)
    # Filter(valid_buffer)

    # Load image (grayscale uint8)
    image = et_obj.get_image(mode="L")

    # Call(lambda image: print(image.dtype), image)

    bg = Call(np.median, image) + 8

    # Call(print, bg)

    mask = image > bg

    # Remove empty images
    any_object = Call(np.any, mask)
    # Call(print, any_object)
    Filter(any_object)

    # mask_fn = Format("{}/{}.png", mask_path, et_obj.object_id)
    # ImageWriter(mask_fn, mask)

    regionprops = ImageProperties(mask, image)
    object_meta = CalculateZooProcessFeatures(
        regionprops, {"id": et_obj.object_id}, "mc_"
    )

    Progress(Format("Process {}", et_obj.object_id), unit_scale=True)

    EcotaxaWriter(features_fn, [], object_meta=object_meta)


p.run()
