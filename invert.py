import concurrent.futures
import fnmatch
import glob
import io
import os
import queue
import shutil
import sys
import threading
import time
import zipfile
from typing import Any, Union
import numpy as np

import PIL.Image
import PIL.ImageOps
import tqdm
from joblib import Parallel, delayed

PIL.Image.init()
PIL_EXTENSIONS = PIL.Image.registered_extensions()

# sys.exit(0)

# 93.2/58, 93.2/80, 93.2/82

PROJECT_IDS = [
    # 368,  # LOKI_PS93.2/58
    362,  # LOKI_PS93.2/80
    # 376,  # LOKI_PS93.2/82
]


def load_member(zf, member):
    fdst = io.BytesIO()
    with zf.open(member) as fsrc:
        shutil.copyfileobj(fsrc, fdst)
    fdst.name = member
    return fdst


def process_image(fsrc, format):
    image = PIL.Image.open(fsrc).convert("L")

    image_np = np.array(image, copy=False)
    background = np.median(image_np)

    # Invert only if white background
    if background > 127:
        image = PIL.ImageOps.invert(image)
        fdst = io.BytesIO()
        image.save(fdst, format=format)
        return fdst
    else:
        name = getattr(fsrc, "name", "n/a")
        print(f"{name} already white-on-black")
        return fsrc


def filter_extension(fns, allowed_extensions):
    return [fn for fn in fns if os.path.splitext(fn)[1] in allowed_extensions]


def parallel_imap(func, iter, max_workers=None):
    if max_workers is None:
        max_workers = os.cpu_count()
    q = queue.Queue(max_workers)
    sentinel = object()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:

        def _loader():
            for obj in iter:
                q.put(ex.submit(func, *obj))
            q.put(sentinel)

        threading.Thread(target=_loader, daemon=True).start()

        while True:
            obj = q.get()
            if obj is sentinel:
                break

            yield obj.result()


for project_id in tqdm.tqdm(PROJECT_IDS, desc="Project...", unit_scale=True):
    source_project_fn = glob.glob(f"black-on-white/export_{project_id}*.zip")[0]
    target_project_fn = os.path.join("input", os.path.basename(source_project_fn))

    print(f"Converting from {source_project_fn} to {target_project_fn}...")

    if os.path.isfile(target_project_fn):
        os.rename(target_project_fn, target_project_fn + ".bak")

    with zipfile.ZipFile(source_project_fn, "r") as zin, zipfile.ZipFile(
        target_project_fn,
        "w",
    ) as zout:
        filenames = zin.namelist()

        image_filenames = filter_extension(filenames, PIL_EXTENSIONS)
        tsv_filenames = fnmatch.filter(filenames, "*.tsv")

        date_time = time.localtime(time.time())[:6]

        #  Copy over TSVs
        for fn in tqdm.tqdm(tsv_filenames, desc="Copying TSV...", unit_scale=True):
            zinfo = zipfile.ZipInfo(filename=fn, date_time=date_time)
            # Text data can be compressed
            zinfo.compress_type = zipfile.ZIP_DEFLATED

            with zin.open(fn, "r") as fin, zout.open(zinfo, "w") as fout:
                shutil.copyfileobj(fin, fout)

        def _loader():
            for fn in tqdm.tqdm(
                image_filenames, desc="Converting images...", unit_scale=True
            ):
                fsrc = load_member(zin, fn)

                yield fn, fsrc

        def _processor(fn, fsrc):
            img_ext = os.path.splitext(fn)[1]
            pil_format = PIL_EXTENSIONS[img_ext]
            return fn, process_image(fsrc, pil_format)

        # results = Parallel(n_jobs=-1)(
        #     delayed(_processor)(fn, fsrc) for fn, fsrc in _loader()
        # )

        results = parallel_imap(_processor, _loader())

        for fn, f_processed in results:
            zinfo = zipfile.ZipInfo(filename=fn, date_time=date_time)
            # Image data is not compressible
            zinfo.compress_type = zipfile.ZIP_STORED

            # Write processed image to output archive
            with zout.open(zinfo, "w") as fout:
                # Rewind file
                f_processed.seek(0)

                # Write to zip
                shutil.copyfileobj(f_processed, fout)

            # print(f"Wrote {fn}.")
