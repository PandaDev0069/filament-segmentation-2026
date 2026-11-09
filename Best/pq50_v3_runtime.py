"""Train-only, date-grouped, instance-balanced segmentation for MAGFiLO Kaggle.

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
VERSION = "filament-instance-balanced-v3"
INPUT_ROOT = Path("/kaggle/input")
WORKER_SCRIPT = Path("/kaggle/working/filament_v3_worker.py")


def default_config():
    return dict(mode="full", data_root="", output_root="/kaggle/working",
        run_name="filament_instance_balanced_v3", gpu_ids=[0, 1], cpu_workers=4,
        seed=20261103, refit_full=False, calibration_fraction=.12, report_fraction=.12,
        minimum_free_gib=2., pretrained=True,
        dense_models=[dict(encoder="resnet34", crop=768, refine=True),
                      dict(encoder="tu-tf_efficientnetv2_s", crop=768, refine=True)],
        dense_epochs=60, dense_steps=240, dense_batch_per_gpu=2, auto_batch=True,
        maximum_batch_per_gpu=6, dense_lr=3e-4, encoder_lr_scale=.1,
        weight_decay=1e-4, accumulation=1, ema_decay=.99, dense_precision="float32",
        minimum_report_pq=.10, training_plots=True, pq_every=6,
        tile_size=1024, tile_overlap=256, inference_batch=2, tta=4,
        max_dense_candidates=192, quality_folds=5,
        pixel_thresholds=[.25, .35, .45, .55, .65], smoke_sources=3,
        total_hours=9.5, training_hours=5.6, owner_radius=16,
        use_previous_weights=True,
        previous_fingerprint="d7e1902f4a7c9cd244f438b8fe6f238fc669daec919224191e1843a945c0cabe",
        warm_start_checkpoints=[], previous_manifest="",
        refit_hours=1., refit_epochs=12, refit_steps=120,
        detector_conf=0., detector_weights="", detector_size=2048,
        detector_epochs=0, detector_batch_per_gpu=1, detector_lr=.001, max_det=100,
        session_started=time.time())


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


def require_finite(tensor, context, synchronize=False):
    """All training ranks stop together BEFORE backward/step on a bad value."""
    bad = (~torch.isfinite(tensor.detach()).all()).to(dtype=torch.int32)
    if synchronize and dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f"{context}: NaN/Inf detected. No invalid checkpoint or probability cache is accepted.")


def require_finite_state(value, context="checkpoint"):
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            require_finite(value, context)
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_state(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_state(item, f"{context}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"{context}: non-finite metadata")


def quantize_probabilities(probability):
    probability = np.asarray(probability)
    if not np.isfinite(probability).all():
        raise FloatingPointError("NaN/Inf mask probabilities; refusing the uint8 conversion that erased masks in v1.")
    if probability.min() < -1e-6 or probability.max() > 1 + 1e-6:
        raise ValueError("Mask probabilities must lie in [0, 1]")
    return np.rint(np.clip(probability, 0, 1) * 255).astype(np.uint8)


def load_dense_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    require_finite_state(value, str(path))
    if "state" not in value or "spec" not in value:
        raise ValueError(f"Missing dense checkpoint state/spec: {path}")
    return value


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

def normalize_solar(gray):
    """Always normalize. Uncertain geometry disables radial correction only.

    Three uint8 channels: global standardization, bounded radial correction,
    CLAHE of the corrected view. No hard limb clipping or morphology inpainting.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("uint8 grayscale required")
    h, w = gray.shape
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    threshold, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    good, cx, cy, radius = False, w / 2, h / 2, .45 * min(h, w)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            (x, y), (a, b), _ = cv2.fitEllipse(contour)
            good = (100 < x < 156 and 100 < y < 156 and 180 < min(a, b) < 253
                    and max(a, b) / max(min(a, b), 1) < 1.13)
            if good:
                cx, cy = x * w / 256, y * h / 256
                radius = (a + b) * min(h, w) / (4 * 256)
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt(((xx - cx) / radius)**2 + ((yy - cy) / radius)**2)
    support = (rr < .88) & (gray > max(5, threshold * .35))
    if support.sum() < h * w * .1:
        support = (rr < .75)
    values = gray[support].astype(np.float32)
    mean, std = float(values.mean()), max(float(values.std()), 8.)
    raw = np.clip(128 + 32 * (gray.astype(np.float32) - mean) / std, 0, 255)
    corrected = gray.astype(np.float32)
    if good:
        # Statistics are estimated at 256² to avoid repeated full-disk sorting.
        sr = cv2.resize(rr.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
        bins = np.minimum((sr * 48).astype(int), 63)
        profile = np.array([np.median(small[(bins == k) & (small > 5)])
                            if np.any((bins == k) & (small > 5)) else np.nan for k in range(48)])
        valid = np.flatnonzero(np.isfinite(profile))
        if len(valid) > 24:
            profile = np.interp(np.arange(48), valid, profile[valid])
            profile = ndi.gaussian_filter1d(profile, 2)
            reference = float(np.median(profile[5:35]))
            gain = np.clip(reference / np.maximum(profile, 20), .75, 1.35)
            correction = np.interp(np.minimum(rr, 47 / 48) * 48, np.arange(48), gain)
            corrected *= correction.astype(np.float32)
    cvalues = corrected[support]
    corrected = np.clip(128 + 32 * (corrected - float(cvalues.mean())) /
                        max(float(cvalues.std()), 8.), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(corrected)
    return np.stack([raw.astype(np.uint8), corrected, clahe], -1), dict(
        geometry_ok=bool(good), mean=mean, std=std)


def discover_root(configured=""):
    """Read the attached competition data exclusively under /kaggle/input."""
    input_root = INPUT_ROOT.resolve()
    name = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if not input_root.is_dir():
        raise FileNotFoundError("/kaggle/input is missing. Run this notebook on Kaggle and attach the competition data.")
    if configured:
        candidates = [Path(configured).resolve()]
        if not candidates[0].is_relative_to(input_root):
            raise ValueError("DATA_ROOT_OVERRIDE must point inside /kaggle/input.")
    else:
        candidates = sorted({p.parent.parent for p in input_root.rglob(name)})
    matches = [root for root in candidates if
               (root / "train" / name).is_file() and
               (root / "train" / "train_images").is_dir() and
               (root / "test" / "test_images").is_dir()]
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError("Multiple competition copies found. Set DATA_ROOT_OVERRIDE to one of: " +
                         ", ".join(map(str, matches)))
    raise FileNotFoundError("Attach the filament-segmentation-2026 competition dataset using Add Input. "
                            "Expected its train annotation JSON, train/train_images and test/test_images.")


def index_data(root, cfg):
    path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    data = load_json(path)
    annotations = defaultdict(list)
    for a in data["annotations"]:
        annotations[a["image_id"]].append(a)
    by_name = defaultdict(list)
    for row in data["images"]:
        assert (row["height"], row["width"]) == (SIZE, SIZE)
        rles = [annotation_rle(a) for a in annotations[row["id"]]]
        boxes = [list(map(float, mu.toBbox({**r, "counts": r["counts"].encode()}))) for r in rles]
        by_name[Path(row["file_name"]).name].append(dict(id=row["id"], rles=rles, boxes=boxes))
    sources = [dict(name=n, stem=Path(n).stem, date=Path(n).stem[:8], views=v,
                    image=str(root / "train" / "train_images" / n)) for n, v in sorted(by_name.items())]
    dates = np.array(sorted({s["date"] for s in sources}))
    np.random.default_rng(cfg["seed"]).shuffle(dates)
    nr = max(1, round(len(dates) * cfg["report_fraction"]))
    nc = max(1, round(len(dates) * cfg["calibration_fraction"]))
    report, calibration = set(dates[:nr]), set(dates[nr:nr + nc])
    splits = {"train": [], "calibration": [], "report": []}
    for source in sources:
        key = "report" if source["date"] in report else "calibration" if source["date"] in calibration else "train"
        splits[key].append(source)
    tests = [dict(name=p.name, stem=p.stem, date=p.stem[:8], image=str(p), views=[])
             for p in sorted((root / "test" / "test_images").glob("*"))
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if {s["stem"] for s in sources} & {s["stem"] for s in tests}:
        raise ValueError("Train/test filenames overlap.")
    if cfg["mode"] == "smoke":
        splits = {k: v[:cfg["smoke_sources"]] for k, v in splits.items()}
        tests = tests[:cfg["smoke_sources"]]
    splits["test"] = tests
    assert not (set(s["date"] for s in splits["train"]) &
                set(s["date"] for k in ("calibration", "report") for s in splits[k]))
    return splits, file_sha(path)


def prepare(cfg):
    check_deadline(cfg, "data preparation", reserve=600)
    cfg = copy.deepcopy(cfg)
    for computed in ("run", "fingerprint", "annotation_sha", "runtime_sha"):
        cfg.pop(computed, None)
    if cfg["cpu_workers"] != 4:
        raise ValueError("This run reserves exactly four CPU workers in total.")
    if cfg["mode"] not in ("full", "smoke"):
        raise ValueError("mode must be full or smoke")
    if cfg["mode"] == "smoke":
        cfg.update(auto_batch=False, maximum_batch_per_gpu=1, dense_epochs=1, dense_steps=2, pq_every=1, detector_epochs=1,
                   detector_size=512, tta=1, refit_full=False, tile_size=512,
                   tile_overlap=64, dense_batch_per_gpu=1, accumulation=1,
                   max_dense_candidates=32, pixel_thresholds=[.5])
        cfg["dense_models"] = [dict(encoder="resnet18", crop=256)]
        cfg["detector_weights"] = "yolo11n.pt"
    root = discover_root(cfg["data_root"])
    cfg["data_root"] = str(root)
    cfg["output_root"] = str(Path(cfg["output_root"]).resolve())
    splits, annotation_sha = index_data(root, cfg)
    image_hashes = dict(bounded_map(lambda s: (s["stem"], file_sha(s["image"])),
        [s for rows in splits.values() for s in rows], cfg["cpu_workers"]))
    runtime_sha = file_sha(WORKER_SCRIPT)
    fingerprint = digest(dict(version=VERSION, cfg={k: v for k, v in cfg.items() if k != "session_started"}, annotations=annotation_sha,
                              sources={k: [s["name"] for s in v] for k, v in splits.items()},
                              images=image_hashes, runtime=runtime_sha))
    run = Path(cfg["output_root"]).resolve() / (cfg["run_name"] + "_" + fingerprint[:10])
    run.mkdir(parents=True, exist_ok=True)
    cfg.update(run=str(run), fingerprint=fingerprint, annotation_sha=annotation_sha, runtime_sha=runtime_sha)
    all_sources = [s for rows in splits.values() for s in rows]
    # Measure actual compressed cache size, rather than budgeting multiple RGB copies.
    cache = run / "cache"
    cache.mkdir(exist_ok=True)
    sample_sizes = []
    for s in all_sources[:min(4, len(all_sources))]:
        feature, _ = normalize_solar(read_gray(s["image"]))
        ok, png = cv2.imencode(".png", feature, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise OSError("PNG encoder failed")
        sample_sizes.append(len(png))
    # Only missing files need disk headroom on rerun. Targets are small bit-packed PNGs.
    missing = sum(not (cache / (s["stem"] + ".png")).exists() for s in all_sources)
    estimated = (max(sample_sizes) * missing * 1.35 + 200_000 *
                 sum(len(s["views"]) for s in all_sources)) / 2**30
    free = shutil.disk_usage(run).free / 2**30
    required = estimated + 3.5 + cfg["minimum_free_gib"]
    print(f"Disk free {free:.1f} GiB; estimated new cache {estimated:.1f}, models/maps 3.5, "
          f"reserve {cfg['minimum_free_gib']:.1f}. No Ultralytics disk cache or duplicated images.", flush=True)
    if free < required:
        raise OSError(f"Estimated incremental need {required:.1f} GiB, free {free:.1f}. "
                      "Choose an output disk with more room; no dataset is deleted automatically.")

    def process(s):
        check_deadline(cfg, "data preparation", reserve=600)
        s = copy.deepcopy(s)
        s["image_sha"] = image_hashes[s["stem"]]
        target = cache / (s["stem"] + ".png")
        meta_path = cache / (s["stem"] + ".json")
        if not target.is_file() or not meta_path.is_file():
            image, meta = normalize_solar(read_gray(s["image"]))
            if not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError(target)
            save_json(meta_path, meta)
        s.update(cache=str(target), photometry=load_json(meta_path))
        for view in s["views"]:
            packed_path = cache / (view["id"] + "_target.png")
            if not packed_path.is_file():
                fg, edges = np.zeros((SIZE, SIZE), np.uint8), np.zeros((SIZE, SIZE), np.uint8)
                for rle in view["rles"]:
                    mask = decode(rle).astype(np.uint8)
                    fg |= mask
                    edges |= cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
                if not cv2.imwrite(str(packed_path), fg + 2 * edges):
                    raise OSError(packed_path)
            view["target"] = str(packed_path)
        return s

    for name, sources in splits.items():
        splits[name] = list(bounded_map(process, sources, cfg["cpu_workers"]))
        print(f"Prepared {name}: {len(sources)} physical images, "
              f"{sum(len(s['views']) for s in sources)} annotator records", flush=True)
    import importlib.metadata as metadata
    packages = {name: metadata.version(name) for name in
                ("torch", "torchvision", "segmentation-models-pytorch", "timm", "scikit-learn", "pycocotools")}
    manifest = dict(config=cfg, splits=splits, packages=packages, python=sys.version)
    save_json(run / "manifest.json", manifest)
    save_json(run / "config.json", cfg)
    prepare_detector(manifest)
    return cfg, manifest


def prepare_detector(manifest):
    # Retained hook in prepare(); v3 has no detector dependencies or artifacts.
    return None

def device_setup(cfg, distributed=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. Use Kaggle GPU T4 x2.")
    local = int(os.environ.get("LOCAL_RANK", 0))
    gpu = cfg["gpu_ids"][local]
    torch.cuda.set_device(gpu)
    if distributed and len(cfg["gpu_ids"]) > 1:
        dist.init_process_group("gloo" if os.name == "nt" else "nccl", timeout=timedelta(minutes=45))
    return torch.device(f"cuda:{gpu}"), local, len(cfg["gpu_ids"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dense(spec, pretrained=False):
    return RefinedUNet(spec, pretrained)


def train_mode(model):
    model.train()
    for child in model.encoder.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


class CropDataset(Dataset):
    def __init__(self, sources, crop, length, seed):
        self.sources, self.crop, self.length, self.seed, self.epoch = sources, crop, length, seed, 0
        self.cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * 10_000_019 + int(index))
        source = self.sources[int(rng.integers(len(self.sources)))]
        view = source["views"][int(rng.integers(len(source["views"])))]
        if view["id"] not in self.cache:
            self.cache.clear()
            image = cv2.imread(source["cache"])
            packed = cv2.imread(view["target"], cv2.IMREAD_GRAYSCALE)
            owner = cv2.imread(view["owner"], cv2.IMREAD_UNCHANGED)
            if image is None or packed is None or owner is None:
                raise OSError("Missing image/target/owner cache")
            self.cache[view["id"]] = image, packed, owner
        image, packed, owner = self.cache[view["id"]]
        # Source context varies; output pixel grids and test masks stay native-sized.
        size = min(SIZE, max(128, int(round(self.crop * rng.uniform(.70, 1.45) / 32)) * 32))
        if rng.random() < .5:
            size = self.crop  # retain exact original pixel spacing in half of crops
        choice = rng.random()
        if choice < .70 and view["boxes"]:
            boxes = np.asarray(view["boxes"])
            # Uniform instance sampling: small/thin filaments are not area-suppressed.
            x, y, w, h = boxes[int(rng.integers(len(boxes)))]
            cx, cy = x + w * rng.uniform(.2, .8), y + h * rng.uniform(.2, .8)
            x0, y0 = cx - size * rng.uniform(.3, .7), cy - size * rng.uniform(.3, .7)
        elif choice < .85:
            # Sample dark backgrounds inside the disk, rather than the black exterior.
            points = rng.integers(200, SIZE - 200, size=(48, 2))
            valid = ((points - SIZE/2)**2).sum(1) < (SIZE*.39)**2
            points = points[valid]
            if not len(points):
                points = np.array([[SIZE//2, SIZE//2]])
            values = image[points[:, 0], points[:, 1], 1].astype(np.float32)
            values += ((packed[points[:, 0], points[:, 1]] & 1) > 0) * 255
            cy, cx = points[int(np.argmin(values))]
            x0, y0 = cx - size/2, cy - size/2
        else:
            x0, y0 = rng.integers(0, SIZE - size + 1, size=2)
        x0, y0 = int(np.clip(x0, 0, SIZE-size)), int(np.clip(y0, 0, SIZE-size))
        x = image[y0:y0+size, x0:x0+size].copy()
        p = packed[y0:y0+size, x0:x0+size]
        ids = owner[y0:y0+size, x0:x0+size]
        if size != self.crop:
            interpolation = cv2.INTER_AREA if size > self.crop else cv2.INTER_LINEAR
            x = cv2.resize(x, (self.crop, self.crop), interpolation=interpolation)
            p = cv2.resize(p, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
            ids = cv2.resize(ids, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
        y = np.stack([(p & 1) > 0, (p & 2) > 0, ids], -1).astype(np.float32)
        k = int(rng.integers(4))
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < .5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = x.astype(np.float32) / 255
        x = np.clip((x-.5) * rng.uniform(.8, 1.2) + .5 + rng.uniform(-.06, .06), 0, 1)
        if rng.random() < .25:
            x = x ** rng.uniform(.85, 1.15)
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))))


def dense_loss(logits, targets, check=True):
    if targets.shape[1] != 3:
        raise ValueError("v3 requires foreground, contour and independent instance-owner targets")
    if check:
        require_finite(logits, "mask loss logits")
        require_finite(targets, "mask loss targets")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, targets = logits.float(), targets.float()
        fg, edge, owner = targets[:, :1], targets[:, 1:2], targets[:, 2:3]
        probability = logits[:, :1].sigmoid()
        bce = (F.binary_cross_entropy_with_logits(logits[:, :1], fg, reduction="none") * (1+2*edge)).mean()
        axes = (1, 2, 3)
        dice = 1 - ((2*(probability*fg).sum(axes)+1)/(probability.sum(axes)+fg.sum(axes)+1)).mean()
        instance = instance_iou_loss(probability, fg, owner)
        contour = F.binary_cross_entropy_with_logits(logits[:, 1:2], edge,
            pos_weight=logits.new_tensor(3.))
        return .35*bce + .20*dice + .40*instance + .05*contour

def update_ema(ema, model, decay):
    with torch.no_grad():
        for dst, src in zip(ema.parameters(), model.parameters()):
            dst.lerp_(src.detach(), 1 - decay)
        for dst, src in zip(ema.buffers(), model.buffers()):
            dst.copy_(src.detach())


def dense_optimizer(model, cfg):
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    decoder = [p for p in model.parameters() if id(p) not in encoder_ids]
    return torch.optim.AdamW([
        dict(params=encoder, lr_scale=cfg.get("encoder_lr_scale", .1)),
        dict(params=decoder, lr_scale=1.),
    ], lr=cfg["dense_lr"], weight_decay=cfg["weight_decay"])


def dense_step(model, optimizer, ema, updates, decay, synchronize=False):
    # Finite gradients and a finite global norm are required before the optimizer.
    norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
    require_finite(norm, "mask gradient norm before optimizer", synchronize)
    optimizer.step()
    norms = torch.stack(torch._foreach_norm([p.detach() for p in model.parameters()]))
    require_finite(norms, "mask parameters after optimizer", synchronize)
    updates += 1
    update_ema(ema, model, min(decay, (1 + updates) / (10 + updates)))
    optimizer.zero_grad(set_to_none=True)
    return updates


def tile_starts(size, tile, overlap):
    if tile >= size:
        return [0]
    return sorted(set([*range(0, size - tile + 1, tile - overlap), size - tile]))


@torch.inference_mode()
def dense_predict(model, image, device, tile=1024, overlap=256, batch=2, tta=4):
    h, w = image.shape[:2]
    tile = min(tile, h, w)
    if tile % 32 or not 0 <= overlap < tile or tta not in (1, 4, 8):
        raise ValueError("Tiles must be divisible by 32; 0 <= overlap < tile; TTA = 1/4/8")
    positions = [(y, x) for y in tile_starts(h, tile, overlap) for x in tile_starts(w, tile, overlap)]
    hann = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), .02).astype(np.float32)
    total, weight = np.zeros((2, h, w), np.float32), np.zeros((h, w), np.float32)
    model.eval()
    for start in range(0, len(positions), batch):
        pos = positions[start:start + batch]
        values = np.stack([image[y:y + tile, x:x + tile] for y, x in pos]).astype(np.float32) / 255
        values = (values - np.array([.485, .456, .406])) / np.array([.229, .224, .225])
        inp = torch.from_numpy(values.transpose(0, 3, 1, 2).copy()).to(device, dtype=torch.float32)
        probabilities = torch.zeros((len(pos), 2, tile, tile), device=device, dtype=torch.float32)
        for t in range(tta):
            dims = [] if t % 4 == 0 else [-1] if t % 4 == 1 else [-2] if t % 4 == 2 else [-2, -1]
            z = inp.transpose(-1, -2) if t >= 4 else inp
            z = z.flip(dims) if dims else z
            # T4 FP16 activations produced NaNs in the failed run. Use float32
            # for the complete mask forward, not merely the loss or sigmoid.
            with torch.autocast(device_type=device.type, enabled=False):
                logits = model(z.float())
                require_finite(logits, "native mask inference logits")
                p = logits.float().sigmoid()
            p = p.flip(dims) if dims else p
            p = p.transpose(-1, -2) if t >= 4 else p
            probabilities.add_(p, alpha=1 / tta)
        arrays = probabilities.cpu().numpy()
        for (y, x), p in zip(pos, arrays):
            total[:, y:y + tile, x:x + tile] += p * hann
            weight[y:y + tile, x:x + tile] += hann
    result = total / np.maximum(weight, 1e-6)
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite blended mask prediction")
    return result


def metric_counts(gt, predicted):
    if gt and predicted:
        a = [{**r, "counts": r["counts"].encode()} for r in gt]
        b = [{**r, "counts": r["counts"].encode()} for r in predicted]
        iou = np.asarray(mu.iou(a, b, [0] * len(b)))
    else:
        iou = np.zeros((len(gt), len(predicted)))
    hit = iou > .5
    return np.array([iou[hit].sum(), hit.sum(), (~hit.any(0)).sum(), (~hit.any(1)).sum()], np.float64)


def summarize(counts):
    iou, tp, fp, fn = map(float, counts)
    denominator = tp + .5 * (fp + fn)
    return dict(PQ=iou / denominator if denominator else 0., SQ=iou / tp if tp else 0.,
                RQ=tp / denominator if denominator else 0., TP=int(tp), FP=int(fp), FN=int(fn), sum_IoU=iou)


def components(prob, threshold=.5, minimum=20, maximum=None):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((prob >= threshold).astype(np.uint8), 8)
    ids = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum) + 1
    if maximum is not None and len(ids) > maximum:
        sums = np.bincount(labels.ravel(), weights=prob.ravel(), minlength=n)
        means = sums[ids] / stats[ids, cv2.CC_STAT_AREA]
        ids = ids[np.argsort(-means, kind="stable")[:maximum]]
    for k in ids:
        x, y, w, h, area = map(int, stats[k])
        if area >= minimum:
            yield (x, y, x + w, y + h), labels[y:y + h, x:x + w] == k


def full_rle(box, mask, shape=None):
    shape = (SIZE, SIZE) if shape is None else shape
    x0, y0, x1, y1 = box
    if mask.shape != (y1 - y0, x1 - x0):
        raise ValueError("ROI and mask shapes differ")
    if mask.size > shape[0] * shape[1] // 8:
        image = np.zeros(shape, np.uint8)
        image[y0:y1, x0:x1] = mask
        return encode(image)
    # Encode sparse ROIs directly in the GLOBAL column-major index space. Avoid
    # allocating/encoding a 4-million-pixel zero canvas for every tiny candidate.
    cols, rows = np.nonzero(mask.T)
    indices = (cols.astype(np.int64) + x0) * shape[0] + rows + y0
    if not len(indices):
        counts = [shape[0] * shape[1]]
    else:
        breaks = np.flatnonzero(np.diff(indices) != 1)
        starts = indices[np.r_[0, breaks + 1]]
        ends = indices[np.r_[breaks, len(indices) - 1]]
        runs = np.empty(len(starts) * 2, np.int64)
        runs[0::2] = starts - np.r_[-1, ends[:-1]] - 1
        runs[1::2] = ends - starts + 1
        counts = runs.tolist()
        tail = shape[0] * shape[1] - int(ends[-1]) - 1
        if tail:
            counts.append(tail)
    rle = mu.frPyObjects(dict(size=list(shape), counts=counts), *shape)
    return dict(size=list(shape), counts=rle["counts"].decode("ascii"))


def checkpoint_save(path, value):
    require_finite_state(value, str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)

def train_dense(config_path, member, full=False):
    cfg = load_json(config_path)
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    if full:
        cfg.update(dense_epochs=cfg["refit_epochs"] if cfg["mode"]=="full" else 1,
            dense_steps=cfg["refit_steps"] if cfg["mode"]=="full" else 2,
            training_hours=cfg["refit_hours"], dense_lr=cfg["dense_lr"]*.33,
            budget_members=len(selected_dense_ids(cfg)),
            warm_start_checkpoints=[str(Path(cfg["run"])/"selection_models"/f"dense_{i}"/"best.pt")
                                    for i in range(len(cfg["dense_models"]))])
        manifest["splits"]["train"] = (manifest["splits"]["train"]+
            manifest["splits"]["calibration"]+manifest["splits"]["report"])
    device, rank, world = device_setup(cfg, distributed=True)
    torch.backends.cudnn.benchmark = True
    seed_all(cfg["seed"] + member*101)
    spec = cfg["dense_models"][member]
    out = Path(cfg["run"]) / ("full_models" if full else "selection_models") / f"dense_{member}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").is_file():
        load_dense_checkpoint(out / "best.pt")
        return
    last = out / "last_training.pt"
    recovery = torch.load(last, map_location="cpu", weights_only=False) if last.is_file() else None
    if recovery is not None:
        require_finite_state(recovery)
        if recovery["fingerprint"] != cfg["fingerprint"]:
            raise RuntimeError("Resume fingerprint differs from the current run")
    warm = cfg.get("warm_start_checkpoints", [])
    if rank == 0:
        model = build_dense(spec, cfg["pretrained"] and recovery is None and not warm)
    if world > 1:
        dist.barrier()
    if rank != 0:
        model = build_dense(spec, cfg["pretrained"] and recovery is None and not warm)
    if recovery is None and warm:
        checkpoint = load_dense_checkpoint(warm[member])
        if checkpoint["spec"]["encoder"] != spec["encoder"]:
            raise ValueError("Warm-start encoder does not match")
        if any(key.startswith("core.") for key in checkpoint["state"]):
            model.load_state_dict(checkpoint["state"])
        else:
            model.core.load_state_dict(checkpoint["state"])
        del checkpoint
    model = model.to(device)
    if recovery is not None:
        batch, measured, steps = recovery["batch"], 0., recovery["steps"]
    elif cfg["mode"] == "smoke":
        batch, measured = cfg["dense_batch_per_gpu"], .1
        steps = cfg["dense_steps"]
    else:
        batch, measured = probe_batch(model, device, spec, cfg, world)
        # One calibration pass is timed with the actual model/device.
        validation_seconds, count = 0., 0
        if not full:
            model.eval()
            torch.cuda.synchronize(device)
            before = time.monotonic()
            image = cv2.imread(manifest["splits"]["calibration"][rank % len(manifest["splits"]["calibration"])] ["cache"])
            dense_predict(model, image, device, cfg["tile_size"], cfg["tile_overlap"], cfg["inference_batch"], 1)
            torch.cuda.synchronize(device)
            validation_seconds = (time.monotonic()-before) * math.ceil(len(manifest["splits"]["calibration"])/world)
            count = math.ceil(cfg["dense_epochs"]/cfg["pq_every"]) + 1 + int(bool(warm))
        steps = plan_steps(cfg, measured, member, validation_seconds*count)
        planned = torch.tensor(steps, device=device)
        if world > 1:
            dist.all_reduce(planned, op=dist.ReduceOp.MIN)
        steps = int(planned.item())
    resource = dict(member=member, gpu=str(device), gpu_name=torch.cuda.get_device_name(device),
        batch_per_gpu=batch, global_batch=batch*world, crop=spec["crop"], precision="float32",
        measured_forward_backward_seconds=measured, steps_per_epoch=steps,
        epochs=cfg["dense_epochs"], loader_workers=cfg["cpu_workers"]//world,
        requested_steps=cfg["dense_steps"], all_data_refit=full,
        training_sources=len(manifest["splits"]["train"]))
    save_json(out / f"rank_{rank}_resources.json", resource)
    print("Measured training plan:", json.dumps(resource), flush=True)
    sources = manifest["splits"]["train"]
    dataset = CropDataset(sources, spec["crop"], steps*batch*world, cfg["seed"]+member*111)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=cfg["seed"])
    workers = cfg["cpu_workers"]//world
    options = dict(num_workers=workers, pin_memory=True, persistent_workers=False)
    if workers:
        options["prefetch_factor"] = 2
    loader = DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True, **options)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = dense_optimizer(model, cfg)
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    begin, best, best_epoch, updates = 0, -1., 0, 0
    if recovery is not None:
        model.load_state_dict(recovery["model"])
        ema.load_state_dict(recovery["ema"])
        optimizer.load_state_dict(recovery["optimizer"])
        begin, best, best_epoch, updates = recovery["epoch"], recovery["best"], recovery["best_epoch"], recovery["updates"]
        del recovery
    elif world > 1:
        # DDP synchronizes random decoder/refiner initialization from rank zero.
        ema.load_state_dict(model.state_dict())
    training_start = time.time()
    allocation = cfg["training_hours"]*3600/cfg.get("budget_members", len(cfg["dense_models"]))
    reserve_hours = .65 if full else min(2.8, cfg["total_hours"]*.35)
    stop_at = min(training_start+allocation, cfg["session_started"]+(cfg["total_hours"]-reserve_hours)*3600)
    completed = begin
    if warm and begin == 0 and not full:
        counts = torch.tensor(fixed_rule_counts(ema, manifest["splits"]["calibration"][rank::world], device, cfg),
                              device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(counts)
        best = summarize(counts.cpu().numpy())["PQ"]
        if rank == 0:
            checkpoint_save(out/"best.pt", dict(state=ema.state_dict(), spec=spec, epoch=0,
                fingerprint=cfg["fingerprint"], precision="float32", numerics_version=3, checkpoint_PQ=best))
            print(f"Warm-start calibration PQ before fine-tuning: {best:.6f}. This checkpoint remains eligible.", flush=True)
    for epoch in range(begin, cfg["dense_epochs"]):
        started = time.time()
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        train_mode(model)
        total = 0.
        torch.cuda.reset_peak_memory_stats(device)
        for step, (x, y) in enumerate(loader):
            # Time progress also decays the LR when the measured budget is tight.
            progress = max((epoch+step/max(steps, 1))/cfg["dense_epochs"],
                           min(1., (time.time()-training_start)/max(stop_at-training_start, 1)))
            warmup = min(1., (epoch*steps+step+1)/max(steps*2, 1))
            lr = cfg["dense_lr"]*warmup*(.03+.97*(1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups:
                group["lr"] = lr*group["lr_scale"]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(x.float())
            require_finite(logits, "v3 training logits", world > 1)
            loss = dense_loss(logits, y, check=False)
            require_finite(loss, "v3 training loss", world > 1)
            loss.backward()
            updates = dense_step(model, optimizer, ema, updates, cfg["ema_decay"], world > 1)
            total += float(loss.detach())
            if (step+1) % 40 == 0:
                print(f"Member {member} rank {rank}: epoch {epoch+1}/{cfg['dense_epochs']}, "
                      f"step {step+1}/{steps}, loss {total/(step+1):.4f}", flush=True)
        counts = np.zeros(4, np.float64)
        evaluate = not full and (epoch == 0 or (epoch+1) % cfg["pq_every"] == 0 or epoch+1 == cfg["dense_epochs"])
        if evaluate:
            counts = fixed_rule_counts(ema, manifest["splits"]["calibration"][rank::world], device, cfg)
        values = torch.tensor([total/max(steps, 1), *counts], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(values)
        values = values.cpu().numpy()
        pq = summarize(values[1:])["PQ"] if evaluate else None
        improved = evaluate and pq > best
        if improved:
            best, best_epoch = pq, epoch+1
        completed = epoch+1
        if full:
            best_epoch = completed
        if rank == 0:
            checkpoint = dict(state=ema.state_dict(), spec=spec, epoch=completed,
                fingerprint=cfg["fingerprint"], precision="float32", numerics_version=3, all_data_refit=full)
            checkpoint_save(out / "last.pt", checkpoint)
            if improved or full:
                checkpoint_save(out / "best.pt", {**checkpoint, "checkpoint_PQ": None if full else best})
            checkpoint_save(last, dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                epoch=completed, best=best, best_epoch=best_epoch, updates=updates,
                steps=steps, batch=batch, fingerprint=cfg["fingerprint"]))
            log = dict(epoch=completed, loss=float(values[0]/world), fixed_rule_calibration_PQ=pq,
                lr=lr, updates=updates, seconds=round(time.time()-started, 2),
                peak_vram_gib=round(torch.cuda.max_memory_allocated(device)/2**30, 3), batch=batch, steps=steps)
            with (out / "history.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(log, allow_nan=False)+"\n")
            print(json.dumps(log), flush=True)
        # Both ranks stop together. Leave a measured epoch margin for checkpoint I/O.
        stop = torch.tensor(int(time.time() + max(60., time.time()-started)*1.15 > stop_at), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            if rank == 0:
                print("Training allocation reached; exporting the best completed checkpoint.", flush=True)
            break
    if rank == 0:
        save_json(out / "complete.json", dict(best_epoch=best_epoch, fixed_rule_calibration_PQ=None if full else best,
            epochs=completed, requested_epochs=cfg["dense_epochs"], updates=updates,
            checkpoint_sha=file_sha(out / "best.pt"), measured_plan=resource,
            training_seconds=time.time()-training_start))
        last.unlink(missing_ok=True)
    if world > 1:
        dist.barrier()



def command(config_path, task, *args, distributed=False):
    cfg = load_json(config_path)
    script = str(Path(WORKER_SCRIPT).resolve())
    cmd = [sys.executable]
    if distributed and len(cfg["gpu_ids"]) > 1:
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={len(cfg['gpu_ids'])}"]
    cmd += [script, "--config", str(config_path), "--task", task, *map(str, args)]
    print("Launching:", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, start_new_session=os.name != "nt")
    try:
        code = process.wait(timeout=max(1, check_deadline(cfg, task)-60))
        if code:
            raise RuntimeError(f"{task} worker failed with exit status {code}")
    except BaseException:
        terminate_worker(process)
        raise


def train_all(cfg, full=False):
    if full:
        require_report_ready(cfg)
        manifest = load_json(Path(cfg["run"])/"manifest.json")
        prepare_instance_targets(cfg, manifest, include_holdout=True)
    path = Path(cfg["run"]) / "config.json"
    members = selected_dense_ids(cfg) if full else range(len(cfg["dense_models"]))
    for member in members:
        reserve = 900 if full else min(3600, cfg["total_hours"]*900)
        check_deadline(cfg, "start next model", reserve=reserve)
        command(path, "dense", "--member", member, *(["--full"] if full else []), distributed=True)


def selected_dense_ids(cfg, all_members=False):
    ids = list(range(len(cfg["dense_models"])))
    selection = Path(cfg["run"]) / "dense_selection.json"
    if all_members or not selection.is_file():
        return ids
    chosen = load_json(selection)
    if chosen["fingerprint"] != cfg["fingerprint"]:
        raise ValueError("Dense model selection belongs to a different run")
    result = chosen["member_ids"]
    if not result or any(i not in ids for i in result):
        raise ValueError("Invalid selected dense members")
    return result


def model_paths(cfg, full=False, all_members=False):
    root = Path(cfg["run"]) / ("full_models" if full else "selection_models")
    return dict(dense=[str(root/f"dense_{i}"/"best.pt") for i in selected_dense_ids(cfg, all_members)])







def prediction_key(cfg, paths):
    return digest(dict(runtime=VERSION, run=cfg["fingerprint"],
        checkpoints=[file_sha(p) for p in paths["dense"]],
        tile=cfg["tile_size"], overlap=cfg["tile_overlap"], tta=cfg["tta"]))[:16]


def predict_worker(config_path, part, gpu_slot=0, full=False):
    cfg = load_json(config_path)
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    device = torch.device(f"cuda:{cfg['gpu_ids'][gpu_slot]}")
    torch.cuda.set_device(device)
    paths = model_paths(cfg, full, all_members=part == "calibration")
    folder = Path(cfg["run"])/"predictions"/prediction_key(cfg, paths)/part
    folder.mkdir(parents=True, exist_ok=True)
    models = []
    for path in paths["dense"]:
        checkpoint = load_dense_checkpoint(path)
        model = build_dense(checkpoint["spec"])
        model.load_state_dict(checkpoint["state"])
        models.append(model.to(device).eval())
    variant = "avg"
    chosen = Path(cfg["run"])/"selected_pipeline.json"
    if chosen.is_file() and part != "calibration":
        variant = load_json(chosen)["recipe"]["set"].split("_", 1)[0]
    sources = manifest["splits"][part][gpu_slot::len(cfg["gpu_ids"])]
    for i, source in enumerate(sources):
        check_deadline(cfg, f"{part} prediction", reserve=90)
        target = folder/(source["stem"]+".npz")
        if target.is_file():
            with np.load(target) as cache:
                if int(cache["schema"]) != 3 or cache["prob"].shape != (2, SIZE, SIZE):
                    raise ValueError(f"Invalid v3 cache: {target}")
            continue
        image = cv2.imread(source["cache"])
        probability = np.zeros((2, SIZE, SIZE), np.float32)
        individual = []
        for model in models:
            prediction = dense_predict(model, image, device, cfg["tile_size"], cfg["tile_overlap"],
                cfg["inference_batch"], cfg["tta"])
            probability += prediction/len(models)
            if part == "calibration":
                individual.append(quantize_probabilities(prediction))
        with target.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f, prob=quantize_probabilities(probability), schema=np.array(3),
                variant=np.array(variant), member_ids=np.array(selected_dense_ids(cfg, all_members=part=="calibration")),
                member_prob=np.stack(individual) if individual else np.empty(0, np.uint8))
        os.replace(target.with_suffix(".tmp"), target)
        print(f"GPU {device.index}: {part} {i+1}/{len(sources)} {source['stem']}", flush=True)


def predict_part(cfg, part, full=False):
    paths = model_paths(cfg, full, all_members=part=="calibration")
    folder = Path(cfg["run"])/"predictions"/prediction_key(cfg, paths)/part
    processes = []
    try:
        for slot in range(len(cfg["gpu_ids"])):
            cmd = [sys.executable, str(Path(WORKER_SCRIPT).resolve()), "--config", str(Path(cfg["run"])/"config.json"),
                "--task", "predict", "--part", part, "--gpu-slot", str(slot)]
            if full:
                cmd.append("--full")
            processes.append(subprocess.Popen(cmd, start_new_session=os.name!="nt"))
        for p in processes:
            code = p.wait(timeout=max(1, check_deadline(cfg, part)-60))
            if code:
                raise RuntimeError(f"{part} prediction worker failed: {code}")
    finally:
        for p in processes:
            terminate_worker(p)
    expected = {s["stem"]+".npz" for s in load_json(Path(cfg["run"])/"manifest.json")["splits"][part]}
    if {p.name for p in folder.glob("*.npz")} != expected:
        raise RuntimeError("Prediction cache does not cover the complete partition")
    save_json(folder/"provenance.json", dict(models=paths, physical_images=len(expected), schema=3))
    return folder



FEATURES = ["log_area", "mean_probability", "max_probability", "q90_probability", "probability_std",
    "boundary_probability", "fill", "aspect", "width_max", "width_mean", "gray_inside", "local_contrast",
    "radius", "detector_confidence", "tta_support", "is_detection", "box_fill", "perimeter_ratio"]


def mask_features(box, mask, prob, edge, image, confidence=0., support=0., detection=False):
    x0, y0, x1, y1 = box
    p = prob[y0:y1, x0:x1][mask]
    e = edge[y0:y1, x0:x1][mask]
    gray = image[y0:y1, x0:x1, 1].astype(np.float32) / 255
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1][mask]
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~mask
    contrast = float(gray[ring].mean() - gray[mask].mean()) if ring.any() else 0.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    points = np.argwhere(mask)
    height, width = mask.shape
    area = int(mask.sum())
    return [math.log1p(area), float(p.mean()), float(p.max()), float(np.quantile(p, .9)), float(p.std()),
        float(e.mean()), area / (height * width), max(height, width) / max(min(height, width), 1),
        float(distance.max()), float(distance.mean()), float(gray[mask].mean()), contrast,
        float(np.hypot(points[:, 1].mean() + x0 - SIZE/2, points[:, 0].mean() + y0 - SIZE/2) / (SIZE/2)),
        float(confidence), float(support), float(detection), area / (height * width),
        float(perimeter / math.sqrt(max(area, 1)))]


def make_candidate(box, mask, prob, edge, image, confidence=0., support=0., detection=False):
    return dict(rle=full_rle(box, mask), local_rle=encode(mask), box=list(map(int, box)), area=int(mask.sum()),
        features=mask_features(box, mask, prob, edge, image, confidence, support, detection))


def candidate_sets(source, prediction_path, thresholds, maximum=192, member=None):
    with np.load(prediction_path) as cache:
        if int(cache["schema"]) != 3:
            raise ValueError("v3 requires newly computed probability maps")
        arrays = {str(cache["variant"]): cache["prob"].astype(np.float32)/255}
        if cache["member_prob"].size:
            arrays = {"avg": cache["prob"].astype(np.float32)/255}
            arrays.update({f"v{int(i)}": p.astype(np.float32)/255
                           for i, p in zip(cache["member_ids"], cache["member_prob"])})
    image = cv2.imread(source["cache"])
    result = {}
    for variant, (prob, edge) in arrays.items():
        for threshold in thresholds:
            for family in ("dense", "close"):
                binary = (prob >= threshold).astype(np.uint8)
                if family == "close":
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                candidates = []
                for box, mask in components(binary.astype(np.float32), .5, 20, maximum):
                    candidates.append(make_candidate(box, mask, prob, edge, image))
                result[f"{variant}_{family}_{threshold}"] = candidates
    return result


def candidate_targets(source, candidates):
    """Expected credited IoU over independent annotator records, using official >0.5."""
    if not candidates:
        return np.zeros(0, np.float32)
    pred = [{**c["rle"], "counts": c["rle"]["counts"].encode()} for c in candidates]
    result = np.zeros(len(pred), np.float64)
    for view in source["views"]:
        gt = [{**r, "counts": r["counts"].encode()} for r in view["rles"]]
        if gt:
            iou = np.asarray(mu.iou(gt, pred, [0] * len(pred)))
            result += (iou * (iou > .5)).sum(0)
    return (result / max(len(source["views"]), 1)).astype(np.float32)


def build_candidate_cache(cfg, part, folder, refresh=False):
    manifest = load_json(Path(cfg["run"])/"manifest.json")
    profile = digest(dict(version=3, thresholds=cfg["pixel_thresholds"], maximum=cfg["max_dense_candidates"]))[:10]
    out = Path(folder)/("candidates_"+profile)
    out.mkdir(exist_ok=True)
    def process(source):
        check_deadline(cfg, f"{part} candidates", reserve=60)
        path = out/(source["stem"]+".json")
        if path.is_file() and not refresh:
            return path
        sets = candidate_sets(source, Path(folder)/(source["stem"]+".npz"),
            cfg["pixel_thresholds"], cfg["max_dense_candidates"])
        if part == "calibration":
            for candidates in sets.values():
                for c, target in zip(candidates, candidate_targets(source, candidates)):
                    c["target"] = float(target)
        save_json(path, sets)
        return path
    files = list(bounded_map(process, manifest["splits"][part], cfg["cpu_workers"]))
    print(f"{part}: cached candidates for {len(files)} physical images", flush=True)
    return out

def quality_estimator(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=7, learning_rate=.05,
        min_samples_leaf=30, l2_regularization=2., early_stopping=False, random_state=seed)


def fit_quality(cfg, candidate_folder):
    """Date-grouped crossfitted scores for calibration, then fit all calibration for report/test.

    Dense checkpoint selection has already used calibration; these are selection
    scores, not unbiased CV. The separate report partition is never fitted/tuned.
    """
    import joblib
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    sources = manifest["splits"]["calibration"]
    x, y, dates, indices = [], [], [], []
    data = {}
    for source in sources:
        sets = load_json(Path(candidate_folder) / (source["stem"] + ".json"))
        data[source["stem"]] = sets
        for name, candidates in sets.items():
            for j, c in enumerate(candidates):
                x.append(c["features"])
                y.append(c["target"])
                dates.append(source["date"])
                indices.append((source["stem"], name, j))
    if not x:
        raise RuntimeError("No calibration candidates. Train longer or inspect normalization and labels.")
    x, y, dates = np.asarray(x, np.float32), np.asarray(y, np.float32), np.asarray(dates)
    unique = np.array(sorted(set(dates)))
    np.random.default_rng(cfg["seed"]).shuffle(unique)
    folds = min(cfg["quality_folds"], len(unique))
    if folds < 2:
        raise RuntimeError("Quality crossfitting needs at least two independent calibration dates.")
    assignment = {date: i % folds for i, date in enumerate(unique)}
    oof = np.zeros(len(y), np.float32)
    for fold in range(folds):
        valid = np.array([assignment[d] == fold for d in dates])
        model = quality_estimator(cfg["seed"] + fold)
        model.fit(x[~valid], y[~valid])
        oof[valid] = model.predict(x[valid])
    for (stem, name, j), score in zip(indices, oof):
        data[stem][name][j]["quality_oof"] = float(np.clip(score, 0, 1))
    model = quality_estimator(cfg["seed"])
    model.fit(x, y)
    joblib.dump(model, Path(cfg["run"]) / "quality_model.joblib")
    save_json(Path(cfg["run"]) / "quality_audit.json", dict(features=FEATURES,
        training_partition="calibration only", crossfit_group="physical date", folds=folds,
        candidate_rows=len(y), dates=len(unique), target="mean official credited IoU per annotation view",
        crossfit_mae=float(np.abs(oof - y).mean()), report_used=False))
    return model, data


def select_instances(candidates, recipe, quality=None, oof=False):
    if not candidates:
        return []
    tree = isinstance(candidates, ComponentTree)
    score_key = "oof" if oof else id(quality)
    if tree and score_key in candidates.cached_scores:
        scores = candidates.cached_scores[score_key]
    elif recipe["quality"]:
        scores = (np.asarray([c["quality_oof"] for c in candidates]) if oof else
            np.clip(quality.predict(np.asarray([c["features"] for c in candidates], np.float32)), 0, 1))
        if tree:
            candidates.cached_scores[score_key] = scores
    else:
        scores = np.asarray([c["features"][1] for c in candidates])
    if tree:
        parents = candidates.topology()
        children = [[] for _ in candidates]
        for index, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(index)
        def solve(index):
            child_value, child_choice = 0., []
            for child in children[index]:
                value, choice = solve(child)
                child_value += value
                child_choice.extend(choice)
            utility = scores[index]-recipe["cutoff"] if candidates[index]["area"]>=recipe["min_area"] else -1.
            return (float(utility), [index]) if utility>max(0., child_value) else (child_value, child_choice)
        chosen = []
        for root in np.flatnonzero(parents<0):
            chosen.extend(solve(int(root))[1])
        return [candidates[i]["rle"] for i in chosen]
    # Every candidate set is the connected components of ONE map/threshold.
    # Thus masks are already disjoint. Do not allocate a 2048 canvas, re-encode,
    # or run another NMS for every one of the hundreds of calibration recipes.
    return [c["rle"] for c, score in zip(candidates, scores)
            if score >= recipe["cutoff"] and c["area"] >= recipe["min_area"]]


def recipe_candidates(sets, name):
    if not name.endswith("_tree"):
        return sets[name]
    memo = name+"_memo"
    if memo not in sets:
        prefix = name[:-4]  # 'v1_dense_' or 'avg_close_'
        candidates = [c for key, values in list(sets.items())
                      if key.startswith(prefix) and not key.endswith("_memo") for c in values]
        sets[memo] = ComponentTree(candidates)
    return sets[memo]


def recipes(cfg):
    variants = [f"v{i}" for i in range(len(cfg["dense_models"]))] + ["avg"]
    fixed = [dict(set=f"{variant}_{family}_{pixel}", quality=quality, cutoff=cutoff, min_area=area)
        for variant in variants for family in ("dense", "close") for pixel in cfg["pixel_thresholds"]
        for quality in (False, True) for cutoff in ([.15, .25, .35, .45] if quality else [.55, .70, .85])
        for area in (40, 100, 200)]
    trees = [dict(set=f"{variant}_{family}_tree", quality=True, cutoff=cutoff, min_area=area)
        for variant in variants for family in ("dense", "close") for cutoff in (.15, .20, .25, .30)
        for area in (40, 100, 200)]
    return fixed+trees


def calibrate(cfg, candidate_folder):
    check_deadline(cfg, "calibration", reserve=900)
    model, data = fit_quality(cfg, candidate_folder)
    sources = load_json(Path(cfg["run"])/"manifest.json")["splits"]["calibration"]
    candidates = recipes(cfg)
    def score_source(source):
        check_deadline(cfg, "calibration sweep", reserve=600)
        counts, numbers = np.zeros((len(candidates), 4), np.float64), np.zeros(len(candidates), np.int64)
        sets = data[source["stem"]]
        for i, recipe in enumerate(candidates):
            pred = select_instances(recipe_candidates(sets, recipe["set"]), recipe, model, oof=True)
            numbers[i] = len(pred)
            for view in source["views"]:
                counts[i] += metric_counts(view["rles"], pred)
        return counts, numbers
    counts, numbers = np.zeros((len(candidates), 4)), np.zeros(len(candidates), np.int64)
    date_counts = defaultdict(lambda: np.zeros_like(counts))
    for source, (c, n) in zip(sources, bounded_map(score_source, sources, cfg["cpu_workers"])):
        counts += c
        numbers += n
        date_counts[source["date"]] += c
    grid = [dict(**r, **summarize(c), physical_predictions=int(n)) for r, c, n in zip(candidates, counts, numbers)]
    selected_index, comparisons = stable_recipe_choice(cfg, candidates, counts, date_counts)
    selected = grid[selected_index]
    grid.sort(key=lambda x: x["PQ"], reverse=True)
    chosen = {k: selected[k] for k in ("set", "quality", "cutoff", "min_area")}
    variant = chosen["set"].split("_", 1)[0]
    ids = list(range(len(cfg["dense_models"]))) if variant=="avg" else [int(variant[1:])]
    save_json(Path(cfg["run"])/"dense_selection.json", dict(member_ids=ids, fingerprint=cfg["fingerprint"]))
    save_json(Path(cfg["run"])/"calibration_grid.json", grid)
    save_json(Path(cfg["run"])/"calibration_stability.json", dict(comparisons=comparisons,
        bootstrap_group="calibration date", report_used=False, selected=selected))
    save_json(Path(cfg["run"])/"selected_pipeline.json", dict(recipe=chosen, calibration=selected,
        score_type="selection score; crossfitted quality; checkpoint/recipe selection is not independent CV",
        final_quality_fit="calibration dates only; report untouched"))
    print("Frozen pipeline:", json.dumps(selected), flush=True)
    return chosen

def evaluate_report(cfg, candidate_folder, recipe):
    import joblib
    model = joblib.load(Path(cfg["run"]) / "quality_model.joblib")
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    totals, rows, cohorts = np.zeros(4, np.float64), [], defaultdict(lambda: np.zeros(4, np.float64))
    for source in manifest["splits"]["report"]:
        data = load_json(Path(candidate_folder) / (source["stem"] + ".json"))
        pred = select_instances(recipe_candidates(data, recipe["set"]), recipe, model)
        count = np.zeros(4, np.float64)
        for view in source["views"]:
            count += metric_counts(view["rles"], pred)
        totals += count
        cohort = "disk_fit" if source["photometry"]["geometry_ok"] else "fallback_normalized"
        cohorts[cohort] += count
        rows.append(dict(stem=source["stem"], date=source["date"], geometry_ok=source["photometry"]["geometry_ok"],
            annotators=len(source["views"]), predictions=len(pred), **summarize(count)))
    metrics = dict(**summarize(totals), physical_images=len(rows),
        fingerprint=cfg["fingerprint"], recipe=recipe, selected_dense_members=selected_dense_ids(cfg),
        cohorts={k: summarize(v) for k, v in cohorts.items()},
        note="Independent report; frozen recipe and calibration-fitted quality model. Before all-data refit.")
    save_json(Path(cfg["run"]) / "report_metrics.json", metrics)
    save_json(Path(cfg["run"]) / "report_by_image.json", rows)
    print("Held-out report:", json.dumps(metrics, indent=2), flush=True)
    require_report_ready(cfg)
    return metrics


def require_report_ready(cfg):
    if cfg["mode"] == "smoke":
        return
    path = Path(cfg["run"]) / "report_metrics.json"
    if not path.is_file():
        raise RuntimeError("Evaluate the frozen report pipeline before full-data refitting or CSV export.")
    report = load_json(path)
    require_finite_state(report, "report")
    if report.get("fingerprint") != cfg["fingerprint"]:
        raise ValueError("Report belongs to a different run")
    threshold = cfg.get("minimum_report_pq", .1)
    if report["PQ"] < threshold:
        raise RuntimeError(f"Report PQ {report['PQ']:.4f} is below the collapse guard {threshold:.2f}. "
                           "Stopped before refitting/export. Inspect finite training losses and native mask previews.")


def export_submission(cfg, candidate_folder, recipe):
    require_report_ready(cfg)
    import joblib
    model = joblib.load(Path(cfg["run"]) / "quality_model.joblib")
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    if cfg["mode"] == "full" and len(manifest["splits"]["test"]) != 180:
        raise ValueError("Full export requires all 180 competition test images.")
    name = "smoke_submission.csv" if cfg["mode"] == "smoke" else "submission.csv"
    target = Path(cfg["run"]) / name
    temporary = target.with_suffix(".tmp")
    audit, identifiers = [], set()
    with temporary.open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        for source in manifest["splits"]["test"]:
            check_deadline(cfg, "CSV export", reserve=30)
            data = load_json(Path(candidate_folder) / (source["stem"] + ".json"))
            pred = select_instances(recipe_candidates(data, recipe["set"]), recipe, model)
            union = np.zeros((SIZE, SIZE), bool)
            for i, rle in enumerate(pred, 1):
                mask = decode(rle)
                assert mask.shape == (SIZE, SIZE) and mask.any() and not (union & mask).any()
                assert encode(mask)["counts"] == rle["counts"]
                union |= mask
                identifier = f"{source['stem']}_{i}"
                assert identifier not in identifiers
                identifiers.add(identifier)
                writer.writerow([identifier, rle["counts"]])
            audit.append(dict(image=source["name"], predictions=len(pred), geometry_ok=source["photometry"]["geometry_ok"]))
    os.replace(temporary, target)
    assert len(audit) == len(manifest["splits"]["test"])
    if cfg["mode"] == "full" and len(audit) != 180:
        raise ValueError(f"Expected 180 competition test images, found {len(audit)}. Do not submit this file.")
    save_json(Path(cfg["run"]) / "submission_audit.json", dict(file=str(target), sha256=file_sha(target),
        native_shape=[SIZE, SIZE], images=len(audit), masks=len(identifiers), zero_prediction_images=sum(r["predictions"] == 0 for r in audit),
        per_image=audit, overlap_check=True, rle_roundtrip_check=True, recipe=recipe,
        smoke=cfg["mode"] == "smoke", all_data_refit=cfg["refit_full"],
        external_filament_ground_truth=False, training_annotation_sha=cfg["annotation_sha"]))
    # Convenient Kaggle root artifact with the exact requested filename, only for full runs.
    if cfg["mode"] == "full":
        shutil.copy2(target, Path(cfg["output_root"]).resolve() / "submission.csv")
    print(f"Wrote {target}: {len(audit)} images, {len(identifiers)} masks", flush=True)
    return target


def preview(cfg, part, candidate_folder, recipe=None, limit=3):
    import matplotlib.pyplot as plt
    import joblib
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    sources = manifest["splits"][part][:limit]
    model = joblib.load(Path(cfg["run"]) / "quality_model.joblib") if recipe else None
    fig, axes = plt.subplots(len(sources), 3, figsize=(14, 4.5 * len(sources)), squeeze=False)
    for row, source in zip(axes, sources):
        gray = read_gray(source["image"])
        row[0].imshow(gray, cmap="gray")
        row[0].set_title(source["stem"])
        with np.load(Path(candidate_folder).parent / (source["stem"] + ".npz")) as values:
            row[1].imshow(values["prob"][0], cmap="magma", vmin=0, vmax=255)
            row[1].set_title("Native ensemble probability")
        row[2].imshow(gray, cmap="gray")
        if recipe:
            sets = load_json(Path(candidate_folder) / (source["stem"] + ".json"))
            pred = select_instances(recipe_candidates(sets, recipe["set"]), recipe, model)
            colors = plt.get_cmap("tab20")
            for i, rle in enumerate(pred):
                mask = decode(rle)
                row[2].contour(mask, levels=[.5], colors=[colors(i % 20)], linewidths=.8)
            row[2].set_title(f"{len(pred)} separate instances")
        for axis in row: axis.axis("off")
    fig.tight_layout()
    path = Path(cfg["run"]) / f"{part}_predictions.png"
    fig.savefig(path, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["dense", "predict"], required=True)
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--part", default="test")
    parser.add_argument("--gpu-slot", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.task=="dense":
            train_dense(args.config, args.member, args.full)
        else:
            predict_worker(args.config, args.part, args.gpu_slot, args.full)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def deadline_remaining(cfg):
    return cfg["session_started"] + cfg["total_hours"] * 3600 - time.time()

def check_deadline(cfg, stage, reserve=0):
    remaining = deadline_remaining(cfg)
    if remaining < reserve:
        raise TimeoutError(f"{stage}: only {remaining/60:.1f} minutes remain in the "
                           "9.5-hour session budget. Existing artifacts are preserved.")
    return remaining

def configure_previous_weights(cfg):
    """Optionally reuse this user's exact, train-only previous run on Kaggle."""
    cfg = copy.deepcopy(cfg)
    if cfg["mode"] != "full" or not cfg.get("use_previous_weights", False):
        return cfg
    candidates = []
    for path in INPUT_ROOT.rglob("manifest.json"):
        try:
            value = load_json(path)
        except (OSError, ValueError):
            continue
        if value.get("config", {}).get("fingerprint") != cfg["previous_fingerprint"]:
            continue
        checkpoints = [path.parent/"selection_models"/f"dense_{i}"/"best.pt"
                       for i in range(len(cfg["dense_models"]))]
        if all(p.is_file() for p in checkpoints):
            candidates.append((path, checkpoints))
    if not candidates:
        print("Previous notebook output not attached: using generic ImageNet initialization.", flush=True)
        return cfg
    path, checkpoints = candidates[0]
    cfg.update(previous_manifest=str(path), warm_start_checkpoints=[str(p) for p in checkpoints],
        warm_start_sha256=[file_sha(p) for p in checkpoints], dense_lr=min(cfg["dense_lr"], 1.5e-4))
    print("Found previous train-only weights. Exact split/image checks will precede training.", flush=True)
    return cfg

def verify_previous_split(cfg, manifest):
    if not cfg.get("warm_start_checkpoints"):
        return
    previous = load_json(cfg["previous_manifest"])
    if previous["config"]["annotation_sha"] != cfg["annotation_sha"]:
        raise ValueError("Warm-start annotations differ from the current competition training data")
    for part in ("train", "calibration", "report"):
        old = {s["stem"]: s["image_sha"] for s in previous["splits"][part]}
        new = {s["stem"]: s["image_sha"] for s in manifest["splits"][part]}
        if old != new:
            raise ValueError(f"Warm-start {part} partition/image hashes differ: held-out contamination is possible")
    save_json(Path(cfg["run"])/"warm_start_audit.json", dict(
        previous_fingerprint=previous["config"]["fingerprint"], original_train_only_weights=True,
        identical_train_calibration_report_partitions=True, identical_image_hashes=True,
        annotation_sha=cfg["annotation_sha"], checkpoint_sha256=cfg["warm_start_sha256"]))
    print("Warm-start split, image hashes and annotation SHA verified.", flush=True)

def make_owner_map(rles, radius=16, shape=(2048, 2048)):
    """Nearest-instance assignment inside a narrow background collar.

    Original independent annotator masks remain the foreground reference. This
    auxiliary map gives each instance an equal soft-IoU term, including nearby
    background false positives. It never merges different annotators' masks.
    """
    owner = np.zeros(shape, np.uint16)
    best = np.full(shape, np.inf, np.float32)
    for index, rle in enumerate(rles, 1):
        if index >= 65535:
            raise ValueError("Too many instances for the auxiliary uint16 target")
        binary = decode(rle).astype(np.uint8)
        if binary.shape != shape:
            raise ValueError("Annotation dimensions do not match the image")
        ys, xs = np.nonzero(binary)
        if not len(xs):
            continue
        x0, x1 = max(0, int(xs.min()) - radius), min(shape[1], int(xs.max()) + radius + 1)
        y0, y1 = max(0, int(ys.min()) - radius), min(shape[0], int(ys.max()) + radius + 1)
        local = binary[y0:y1, x0:x1]
        distance = cv2.distanceTransform(1 - local, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        previous = best[y0:y1, x0:x1]
        update = (distance <= radius) & (distance < previous)
        owner[y0:y1, x0:x1][update] = index
        previous[update] = distance[update]
    return owner

def prepare_instance_targets(cfg, manifest, include_holdout=False):
    check_deadline(cfg, "instance target preparation", reserve=600)
    def process(source):
        for view in source["views"]:
            target = Path(cfg["run"]) / "cache" / (view["id"] + "_owner.png")
            if not target.is_file():
                owner = make_owner_map(view["rles"], cfg["owner_radius"])
                if not cv2.imwrite(str(target), owner):
                    raise OSError(target)
            view["owner"] = str(target)
        return source
    # Holdout targets are built only AFTER the validated model/report is frozen,
    # if a separate all-data refit has explicitly been enabled.
    parts = ("train", "calibration", "report") if include_holdout else ("train",)
    for part in parts:
        manifest["splits"][part] = list(bounded_map(process,
            manifest["splits"][part], cfg["cpu_workers"]))
    save_json(Path(cfg["run"]) / "manifest.json", manifest)
    print("Prepared independent instance targets; no external labels or test supervision.", flush=True)
    return manifest

def audit_split_leakage(cfg, manifest):
    """Assert physical dates and pixel-identical images cannot leak across splits."""
    rows = [(part, s) for part, sources in manifest["splits"].items() for s in sources]
    def hashed(row):
        part, source = row
        pixels = read_gray(source["image"])
        return dict(part=part, stem=source["stem"], date=source["date"],
                    decoded_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    records = list(bounded_map(hashed, rows, cfg["cpu_workers"]))
    groups = defaultdict(list)
    for row in records:
        groups[row["decoded_sha256"]].append(row)
    collisions = [v for v in groups.values() if len({r["part"] for r in v})>1]
    training_dates = {s["date"] for s in manifest["splits"]["train"]}
    heldout_dates = {s["date"] for p in ("calibration", "report") for s in manifest["splits"][p]}
    save_json(Path(cfg["run"])/"split_leakage_audit.json", dict(
        decoded_images=len(records), cross_partition_pixel_duplicates=collisions,
        train_heldout_date_overlap=sorted(training_dates & heldout_dates), records=records))
    if collisions or training_dates & heldout_dates:
        raise ValueError("Cross-partition duplicate/date leakage detected; inspect split_leakage_audit.json")
    print(f"Split audit passed for {len(records)} decoded physical images.", flush=True)

class RefinedUNet(nn.Module):
    """ImageNet U-Net plus a full-resolution intensity/logit residual head."""
    def __init__(self, spec, pretrained=False):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.core = smp.Unet(encoder_name=spec["encoder"],
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3, classes=2, decoder_channels=(128, 64, 32, 16, 16),
            decoder_attention_type="scse")
        def replace_bn(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.GroupNorm(math.gcd(8, child.num_features), child.num_features))
                else:
                    replace_bn(child)
        replace_bn(self.core.decoder)
        self.core.decoder.blocks[-1].attention1.requires_grad_(False)
        self.refine = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.GroupNorm(4, 16),
            nn.SiLU(), nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 2, 1)) if spec.get("refine", True) else None
        if self.refine is not None:
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    @property
    def encoder(self):
        return self.core.encoder

    @property
    def decoder(self):
        return self.core.decoder

    def forward(self, image):
        logits = self.core(image)
        return logits if self.refine is None else logits + self.refine(torch.cat([image, logits], 1))

def instance_iou_loss(probability, foreground, owners):
    """Equal-weight soft IoU over visible instances, not over foreground pixels.

    Scatter reductions include the assigned local background collar in each union.
    This is a differentiable surrogate; it is not the official discrete PQ metric.
    """
    losses = []
    for p, y, ids in zip(probability, foreground, owners):
        ids = ids.flatten().long()
        p, y = p.flatten(), y.flatten()
        # IDs originate in train annotations and have a small bounded range.
        count = int(ids.max().detach().item()) + 1
        intersection = p.new_zeros(count).scatter_add(0, ids, p*y)
        predicted = p.new_zeros(count).scatter_add(0, ids, p)
        target = p.new_zeros(count).scatter_add(0, ids, y)
        valid = target > 0
        valid[0] = False
        value = 1 - (intersection + 1) / (predicted + target - intersection + 1)
        # Empty crops contribute zero here; BCE still supervises their background.
        losses.append((value * valid).sum() / valid.sum().clamp_min(1))
    return torch.stack(losses).mean()

def plan_steps(cfg, seconds_per_step, member, validation_seconds=0.):
    """Complete a cosine schedule inside a measured training allocation."""
    seconds = cfg["training_hours"] * 3600 / cfg.get("budget_members", len(cfg["dense_models"]))
    available = max(1., seconds - validation_seconds - 300)
    steps = max(8, int(available / max(seconds_per_step * 1.65 * cfg["dense_epochs"], .001)))
    return min(cfg["dense_steps"], steps)

def fixed_rule_counts(model, sources, device, cfg):
    counts = np.zeros(4, np.float64)
    for source in sources:
        image = cv2.imread(source["cache"])
        p = dense_predict(model, image, device, cfg["tile_size"], cfg["tile_overlap"], cfg["inference_batch"], 1)[0]
        rles = [full_rle(box, mask) for box, mask in components(p, .4, 80, cfg["max_dense_candidates"])
                if p[box[1]:box[3], box[0]:box[2]][mask].max() > .8]
        for view in source["views"]:
            counts += metric_counts(view["rles"], rles)
    return counts

def probe_batch(model, device, spec, cfg, world):
    """Measure real FP32 forward/backward; both ranks use the smaller safe batch."""
    initial = cfg["dense_batch_per_gpu"]
    candidates = [initial] if not cfg["auto_batch"] else list(range(cfg["maximum_batch_per_gpu"], 0, -1))
    model.eval()
    result, seconds = None, None
    for batch in candidates:
        x = y = z = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.zeros((batch, 3, spec["crop"], spec["crop"]), device=device)
            y = torch.zeros_like(x)
            y[:, 0, 40:80, 40:120] = 1
            y[:, 2, 30:90, 30:130] = 1
            timings = []
            for trial in range(3):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                before = time.monotonic()
                z = model(x)
                loss = dense_loss(z, y)
                loss.backward()
                torch.cuda.synchronize(device)
                timings.append(time.monotonic()-before)
            peak = torch.cuda.max_memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            # Leave room for optimizer state, DDP buckets, EMA and actual variation.
            if peak > .74 * total and cfg["auto_batch"] and batch > 1:
                continue
            result, seconds = batch, float(np.median(timings[1:]))
            break
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
        finally:
            model.zero_grad(set_to_none=True)
            del x, y, z, loss
            gc.collect()
            torch.cuda.empty_cache()
    if result is None:
        raise RuntimeError("No native crop batch fit in the GPU")
    values = torch.tensor([float(result), float(seconds)], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(values[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(values[1:], op=dist.ReduceOp.MAX)
    return int(values[0].item()), float(values[1].item())

def terminate_worker(process):
    if process.poll() is not None:
        return
    if os.name != "nt":
        import signal
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    else:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            import signal
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()

class ComponentTree(list):
    """Nested components across probability thresholds for one model/family.

    Threshold sets (including their monotone closing) form an inclusion forest.
    A dynamic program can choose one complete parent or several disjoint children
    without cutting masks, fabricating pixels, or allowing duplicate ownership.
    """
    def __init__(self, candidates):
        seen = set()
        unique = []
        for candidate in candidates:
            key = candidate["rle"]["counts"]
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        super().__init__(unique)
        self.parents = None
        self.cached_scores = {}

    def topology(self):
        if self.parents is not None:
            return self.parents
        n = len(self)
        parents = np.full(n, -1, np.int32)
        if n:
            rles = [{**c["rle"], "counts": c["rle"]["counts"].encode()} for c in self]
            area = np.asarray([c["area"] for c in self], np.float64)
            iou = np.asarray(mu.iou(rles, rles, [0]*n), np.float64)
            intersection = iou*(area[:, None]+area[None, :])/(1+iou)
            for child in range(n):
                possible = np.flatnonzero((area>area[child]) & (intersection[:, child] >= area[child]-.01))
                if len(possible):
                    parents[child] = possible[np.argmin(area[possible])]
        self.parents = parents
        return parents

def paired_pq_interval(primary, alternative, seed, repeats=500):
    """Date-paired bootstrap of a calibration PQ difference, never report labels."""
    primary, alternative = np.asarray(primary), np.asarray(alternative)
    if primary.shape != alternative.shape or primary.ndim != 2 or primary.shape[1] != 4:
        raise ValueError("Expected aligned [date, four pooled metric counts] arrays")
    rng = np.random.default_rng(seed)
    difference = []
    for _ in range(repeats):
        sample = rng.integers(len(primary), size=len(primary))
        difference.append(summarize(primary[sample].sum(0))["PQ"]-
                          summarize(alternative[sample].sum(0))["PQ"])
    return list(map(float, np.quantile(difference, [.025, .975])))

def stable_recipe_choice(cfg, candidates, counts, date_counts):
    """Prefer averaging/plain components when their disadvantage is uncertain."""
    pq = np.asarray([summarize(c)["PQ"] for c in counts])
    chosen = int(pq.argmax())
    comparisons = []
    dates = np.stack(list(date_counts.values()))
    def prefer(alternative, reason):
        nonlocal chosen
        interval = paired_pq_interval(dates[:, chosen], dates[:, alternative], cfg["seed"])
        use_alternative = interval[0] <= 0
        comparisons.append(dict(numerical_winner=candidates[chosen], alternative=candidates[alternative],
            pq_difference=float(pq[chosen]-pq[alternative]), date_paired_95ci=interval,
            chose_alternative=use_alternative, reason=reason))
        if use_alternative:
            chosen = alternative
    average = [i for i, recipe in enumerate(candidates) if recipe["set"].startswith("avg_")]
    if average and not candidates[chosen]["set"].startswith("avg_"):
        prefer(max(average, key=lambda i: pq[i]), "Prefer the model average unless the single-model gain is reliable across dates")
    variant = candidates[chosen]["set"].split("_", 1)[0]
    if "_close_" in candidates[chosen]["set"] or candidates[chosen]["set"].endswith("_tree"):
        plain = [i for i, recipe in enumerate(candidates) if recipe["set"].startswith(variant+"_dense_")
                 and not recipe["set"].endswith("_tree")]
        if plain:
            prefer(max(plain, key=lambda i: pq[i]), "Prefer unchanged connected components unless grouping/morphology gains are reliable")
    return chosen, comparisons

def add_report_diagnostics(cfg, candidate_folder, recipe):
    """Bootstrap dates and report misses by size without tuning on the report."""
    import joblib
    model = joblib.load(Path(cfg["run"])/"quality_model.joblib")
    sources = load_json(Path(cfg["run"])/"manifest.json")["splits"]["report"]
    dates = defaultdict(lambda: np.zeros(4, np.float64))
    areas = {"under_500": [0, 0], "500_to_1999": [0, 0], "2000_plus": [0, 0]}
    for s in sources:
        data = load_json(Path(candidate_folder)/(s["stem"]+".json"))
        pred = select_instances(recipe_candidates(data, recipe["set"]), recipe, model)
        for view in s["views"]:
            dates[s["date"]] += metric_counts(view["rles"], pred)
            gt = [{**r, "counts": r["counts"].encode()} for r in view["rles"]]
            pr = [{**r, "counts": r["counts"].encode()} for r in pred]
            hit = (np.asarray(mu.iou(gt, pr, [0]*len(pr)))>.5).any(1) if gt and pr else np.zeros(len(gt), bool)
            for r, matched in zip(gt, hit):
                area = int(mu.area(r))
                key = "under_500" if area<500 else "500_to_1999" if area<2000 else "2000_plus"
                areas[key][0] += int(matched)
                areas[key][1] += 1
    counts = np.asarray(list(dates.values()))
    rng = np.random.default_rng(cfg["seed"])
    scores = [summarize(counts[rng.integers(len(counts), size=len(counts))].sum(0))["PQ"] for _ in range(1000)]
    report = load_json(Path(cfg["run"])/"report_metrics.json")
    report.update(date_bootstrap_95ci=list(map(float, np.quantile(scores, [.025, .975]))),
        recall_by_area={k: dict(matched=m, total=n, recall=m/n if n else 0.) for k, (m, n) in areas.items()},
        total_session_seconds=time.time()-cfg["session_started"],
        target_public_pq=.5, target_achieved_on_leaderboard="unverified")
    save_json(Path(cfg["run"])/"report_metrics.json", report)
    return report

def finish_submission_run(cfg, test_candidates, recipe):
    """Keep the audited validated CSV before an optional separately timed refit."""
    run = Path(cfg["run"])
    smoke = cfg["mode"]=="smoke"
    prefix = "smoke_" if smoke else ""
    validated_cfg = {**cfg, "refit_full": False}
    original = export_submission(validated_cfg, test_candidates, recipe)
    validated = run/(prefix+"submission_validated.csv")
    shutil.copy2(original, validated)
    audit = load_json(run/"submission_audit.json")
    audit["file"] = str(validated)
    save_json(run/"validated_submission_audit.json", audit)
    variants = dict(validated=dict(file=str(validated), sha256=file_sha(validated), all_data_refit=False),
                    refit=None, default_file=str(original))
    if cfg.get("refit_full", False):
        needed = cfg["refit_hours"]*3600 + (.65*3600 if not smoke else 60)
        if deadline_remaining(cfg) < needed:
            variants["refit_skipped"] = "Insufficient measured session time; the validated CSV is preserved"
            print(variants["refit_skipped"], flush=True)
        else:
            train_all(cfg, full=True)
            full_maps = predict_part(cfg, "test", full=True)
            full_candidates = build_candidate_cache(cfg, "test", full_maps)
            original = export_submission({**cfg, "refit_full": True}, full_candidates, recipe)
            refit = run/(prefix+"submission_refit.csv")
            shutil.copy2(original, refit)
            audit = load_json(run/"submission_audit.json")
            audit["file"] = str(refit)
            save_json(run/"refit_submission_audit.json", audit)
            variants["refit"] = dict(file=str(refit), sha256=file_sha(refit), all_data_refit=True,
                report_PQ=None, note="Report score belongs to the earlier validated models")
    save_json(run/"submission_variants.json", variants)
    return original, variants

if __name__ == "__main__":
    main()
