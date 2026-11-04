%%filament_section imports_and_io
"""Train-only, date-grouped detector + native mask learning for MAGFiLO Kaggle.

All code is included in this Kaggle notebook. No external filament annotations,
precomputed submissions, or test-derived supervision are used.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import copy
import csv
import gc
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mu
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

cv2.setNumThreads(1)
torch.set_num_threads(1)
SIZE = 2048
VERSION = "kaggle-native-hybrid-v1"
INPUT_ROOT = Path("/kaggle/input")
WORKER_SCRIPT = Path("/kaggle/working/filament_gpu_worker.py")


def default_config():
    return dict(mode="full", data_root="", output_root="/kaggle/working", run_name="kaggle_native_hybrid_v1",
        gpu_ids=[0, 1], cpu_workers=4, seed=20261103, refit_full=True,
        calibration_fraction=.12, report_fraction=.12, minimum_free_gib=2.,
        dense_models=[dict(encoder="resnet34", crop=512),
                      dict(encoder="tu-tf_efficientnetv2_s", crop=768)],
        pretrained=True, dense_epochs=32, dense_steps=288, dense_batch_per_gpu=2,
        dense_lr=3e-4, weight_decay=1e-4, accumulation=3, ema_decay=.995,
        pq_every=4, tile_size=1024, tile_overlap=256, inference_batch=2,
        tta=4, detector_weights="yolo11m.pt", detector_size=2048,
        detector_epochs=45, detector_batch_per_gpu=1, detector_lr=.001,
        detector_conf=.07, max_det=100, max_dense_candidates=512, quality_folds=5,
        pixel_thresholds=[.35, .5, .65], smoke_sources=3)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounded_map(fn, items, workers=4):
    """At most four active/pending native images; never queue the entire dataset."""
    items = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            item = next(items, None)
            if item is not None:
                pending.append(pool.submit(fn, item))
        while pending:
            yield pending.popleft().result()
            item = next(items, None)
            if item is not None:
                pending.append(pool.submit(fn, item))


def encode(mask):
    rle = mu.encode(np.asfortranarray(mask, dtype=np.uint8))
    return dict(size=[int(x) for x in rle["size"]], counts=rle["counts"].decode("ascii"))


def decode(rle):
    rle = dict(rle)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mu.decode(rle).astype(bool)


def annotation_rle(ann, shape=(SIZE, SIZE)):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        r = mu.merge(mu.frPyObjects(seg, *shape))
    elif isinstance(seg["counts"], list):
        r = mu.frPyObjects(seg, *shape)
    else:
        r = dict(seg)
        if isinstance(r["counts"], str):
            r["counts"] = r["counts"].encode("ascii")
    return dict(size=list(shape), counts=r["counts"].decode("ascii"))


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (SIZE, SIZE):
        raise ValueError(f"Expected a readable native {SIZE}x{SIZE} image: {path}")
    return image