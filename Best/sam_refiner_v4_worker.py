from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import argparse,copy,csv,gc,hashlib,json,math,random,shutil,subprocess,sys,time,types
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
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
from torch.utils.data import DataLoader,Dataset,DistributedSampler
cv2.setNumThreads(1)
torch.set_num_threads(1)
SIZE=2048
INPUT_ROOT=Path("/kaggle/input")

FEATURES = ["log_area", "mean_probability", "max_probability", "q90_probability", "probability_std",
    "boundary_probability", "fill", "aspect", "width_max", "width_mean", "gray_inside", "local_contrast",
    "radius", "detector_confidence", "tta_support", "is_detection", "box_fill", "perimeter_ratio"]

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

def update_ema(ema, model, decay):
    with torch.no_grad():
        for dst, src in zip(ema.parameters(), model.parameters()):
            dst.lerp_(src.detach(), 1 - decay)
        for dst, src in zip(ema.buffers(), model.buffers()):
            dst.copy_(src.detach())

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

def quality_estimator(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=7, learning_rate=.05,
        min_samples_leaf=30, l2_regularization=2., early_stopping=False, random_state=seed)
b=types.SimpleNamespace(**{name:globals()[name] for name in ['FEATURES', 'INPUT_ROOT', 'annotation_rle', 'bounded_map', 'candidate_targets', 'checkpoint_save', 'components', 'decode', 'device_setup', 'digest', 'discover_root', 'encode', 'file_sha', 'full_rle', 'load_json', 'make_candidate', 'mask_features', 'metric_counts', 'normalize_solar', 'quality_estimator', 'read_gray', 'require_finite', 'require_finite_state', 'save_json', 'seed_all', 'summarize', 'update_ema']})

"""Train-only SAM crop refinement of the user's frozen solar-filament predictions.

The generic SAM image/prompt encoders are frozen. Only its mask decoder is fitted
to the supplied training annotations. Shared scoring/I/O are embedded by the
notebook builder; this module's local import is not an extra Kaggle input.
"""

import copy
import csv
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mu
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


SAM_WORKER = Path("/kaggle/working/sam_refiner_v4_worker.py")
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
REFINER_VERSION = "sam-crop-refiner-v4"
EXTRA_FEATURES = ["sam_quality", "sam_mask_probability", "seed_coverage", "is_refined"]


def refiner_config():
    return dict(mode="full", input_root="/kaggle/input", output_root="/kaggle/working",
        gpu_ids=[0, 1], cpu_workers=4, seed=20261109, session_started=time.time(),
        total_hours=3.4, cache_hours=.85, train_hours=.85, epochs=24,
        steps=200, batch_per_gpu=16, encode_batch=2, lr=8e-5, weight_decay=.01,
        ema_decay=.99, validation_every=3, validation_records=128,
        max_proposals=32, proposal_thresholds=[.20, .40], mask_thresholds=[.35, .50, .65],
        feature_storage="float16",
        minimum_seed_area=40, roi_context=1.7, minimum_roi=192, maximum_roi=1536,
        quality_folds=5, minimum_calibration_gain=.01, smoke_sources=3,
        prior_fingerprint="90b4b0b58b0caf5353dc319ca37d87d92998e19010a8cf5a2325cbbc38f31668")


def sam_remaining(cfg, reserve=0):
    seconds = cfg["session_started"]+cfg["total_hours"]*3600-time.time()
    if seconds < reserve:
        raise TimeoutError(f"SAM refinement has {seconds/60:.1f} minutes remaining; preserving artifacts")
    return seconds


def square_roi(box, cfg):
    x0, y0, x1, y1 = map(float, box)
    side = int(math.ceil(max(x1-x0, y1-y0)*cfg["roi_context"]/16)*16)
    side = min(2048, cfg["maximum_roi"], max(cfg["minimum_roi"], side))
    left = int(np.clip(round((x0+x1-side)/2), 0, 2048-side))
    top = int(np.clip(round((y0+y1-side)/2), 0, 2048-side))
    return [left, top, left+side, top+side]


def discover_prior(cfg):
    matches = []
    for path in Path(cfg["input_root"]).rglob("manifest.json"):
        try:
            manifest = b.load_json(path)
        except (OSError, ValueError):
            continue
        if manifest.get("config", {}).get("fingerprint")==cfg["prior_fingerprint"]:
            matches.append((path, manifest))
    if len(matches)!=1:
        raise ValueError("Attach output of pandadev0069/filament-v3-warm-start-pilot, containing the 90b4b0b58b run")
    return matches[0]


def prior_crossfit_scores(cfg,prior_root,prior):
    """Reproduce prior date-OOF quality scores for HONEST calibration proposals.

    The old final regressor saw all calibration dates. Using its fitted scores
    to prefilter new calibration proposals would leak their labels into selection.
    Report/test continue to use that final regressor, as in the original pipeline.
    """
    recipe=b.load_json(prior_root/"selected_pipeline.json")["recipe"]
    if not recipe["quality"]:return {}
    folders=list((prior_root/"predictions").glob("*/calibration/candidates_*"))
    folder=next((p for p in folders if all((p/(s["stem"]+".json")).is_file() for s in prior["splits"]["calibration"])),None)
    if folder is None:raise ValueError("The full prior calibration candidate cache is needed for date-OOF reconstruction")
    x=[];y=[];dates=[];indices=[];result={}
    for source in prior["splits"]["calibration"]:
        sets=b.load_json(folder/(source["stem"]+".json"))
        result[source["stem"]]=[0.]*len(sets[recipe["set"]])
        for name,values in sets.items():
            for j,c in enumerate(values):
                x.append(c["features"]);y.append(c["target"]);dates.append(source["date"])
                indices.append((source["stem"],name,j))
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);dates=np.asarray(dates)
    seed=prior["config"]["seed"]
    unique=np.array(sorted(set(dates)));np.random.default_rng(seed).shuffle(unique)
    folds=min(prior["config"]["quality_folds"],len(unique));assignment={date:i%folds for i,date in enumerate(unique)}
    scores=np.zeros(len(y),np.float32)
    for fold in range(folds):
        valid=np.asarray([assignment[d]==fold for d in dates]);model=b.quality_estimator(seed+fold)
        model.fit(x[~valid],y[~valid]);scores[valid]=np.clip(model.predict(x[valid]),0,1)
    for (stem,name,j),score in zip(indices,scores):
        if name==recipe["set"]:result[stem][j]=float(score)
    b.save_json(Path(cfg["run"])/"prior_quality_oof.json",dict(scores=result,seed=seed,folds=folds,
        group="prior calibration date",fitted_prior_quality_not_used_for_calibration_proposal_filter=True))
    return result


def prepare_refiner(cfg):
    cfg = copy.deepcopy(cfg)
    if cfg["mode"] not in ("full", "smoke"):
        raise ValueError("MODE must be full or smoke")
    if cfg["mode"]=="smoke":
        cfg.update(epochs=1, steps=2, batch_per_gpu=2, max_proposals=2,
                   mask_thresholds=[.5], validation_records=4, validation_every=1)
    prior_path, prior = discover_prior(cfg)
    import sklearn
    required=prior.get("packages",{}).get("scikit-learn")
    if required and sklearn.__version__!=required:
        raise RuntimeError(f"Prior quality model requires scikit-learn {required}; found {sklearn.__version__}. Use the notebook setup in a fresh kernel.")
    cfg["sklearn_version"]=sklearn.__version__
    # Verify the provided train JSON, rather than trusting cached annotation metadata.
    original_input = b.INPUT_ROOT
    b.INPUT_ROOT = Path(cfg["input_root"])
    try:
        data_root = b.discover_root("")
    finally:
        b.INPUT_ROOT = original_input
    annotation = data_root/"train"/"MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    if b.file_sha(annotation)!=prior["config"]["annotation_sha"]:
        raise ValueError("Attached annotation JSON differs from the trained prediction source")
    assets = Path(cfg["output_root"])
    assets.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(cfg.get("sam_checkpoint",assets/"sam_vit_b_01ec64.pt"))
    if not checkpoint.is_file():
        torch.hub.download_url_to_file(SAM_URL, str(checkpoint), progress=False)
    cfg["sam_checkpoint"] = str(checkpoint)
    cfg["sam_sha256"] = b.file_sha(checkpoint)
    cfg["prior_root"] = str(prior_path.parent)
    cfg["annotation_sha"] = b.file_sha(annotation)
    cfg["fingerprint"] = b.digest(dict(version=REFINER_VERSION,
        config={k:v for k,v in cfg.items() if k!="session_started"}, worker=b.file_sha(SAM_WORKER)))
    run = assets/("filament_sam_refiner_v4_"+cfg["fingerprint"][:10])
    run.mkdir(exist_ok=True)
    cfg["run"] = str(run)
    prior_oof=prior_crossfit_scores(cfg,prior_path.parent,prior)
    manifest = dict(splits=copy.deepcopy(prior["splits"]), config=cfg,
        prior_calibration=b.load_json(prior_path.parent/"selected_pipeline.json"),
        prior_report=b.load_json(prior_path.parent/"report_metrics.json"))
    images_to_check = []
    for part, sources in manifest["splits"].items():
        if cfg["mode"]=="smoke":
            sources = sources[:cfg["smoke_sources"]]
            manifest["splits"][part] = sources
        for source in sources:
            source["image"] = str(data_root/("test" if part=="test" else "train")/
                ("test_images" if part=="test" else "train_images")/source["name"])
            source["cache"] = str(prior_path.parent/"cache"/(source["stem"]+".png"))
            if not Path(source["cache"]).is_file():
                fallback = run/"images"/(source["stem"]+".png")
                fallback.parent.mkdir(exist_ok=True)
                image, _ = b.normalize_solar(b.read_gray(source["image"]))
                if not cv2.imwrite(str(fallback), image):
                    raise OSError(fallback)
                source["cache"] = str(fallback)
            images_to_check.append(source)
        if part in ("calibration", "report", "test"):
            folders = list((prior_path.parent/"predictions").glob("*/"+part))
            candidates = [p for p in folders if all((p/(s["stem"]+".npz")).is_file() for s in sources)]
            if not candidates:
                raise ValueError(f"Missing frozen {part} prediction maps in the attached prior output")
            for source in sources:
                source["prior_map"] = str(candidates[0]/(source["stem"]+".npz"))
                if part=="calibration" and source["stem"] in prior_oof:
                    source["prior_oof"] = prior_oof[source["stem"]]
                old_candidates = list(candidates[0].glob("candidates_*/"+source["stem"]+".json"))
                if not old_candidates:
                    raise ValueError("The prior's exact candidate cache is required for the baseline comparison")
                source["prior_candidates"] = str(old_candidates[0])
    for source, actual in zip(images_to_check, b.bounded_map(lambda s:b.file_sha(s["image"]), images_to_check,4)):
        if actual != source["image_sha"]:
            raise ValueError("Raw image hash mismatch against the frozen prediction source")
    train_dates = {s["date"] for s in manifest["splits"]["train"]}
    heldout_dates = {s["date"] for part in ("calibration", "report") for s in manifest["splits"][part]}
    if train_dates & heldout_dates:
        raise ValueError("Training/held-out dates overlap")
    b.save_json(run/"config.json",cfg)
    b.save_json(run/"manifest.json",manifest)
    b.save_json(run/"input_audit.json",dict(annotation_sha=cfg["annotation_sha"],
        image_hashes_verified=len(images_to_check), train_heldout_dates_disjoint=True,
        generic_sam_checkpoint_sha=cfg["sam_sha256"], external_solar_ground_truth=False))
    print("Prepared SAM refinement:",run,flush=True)
    return cfg, manifest


def grouped_records(source, part, cfg):
    views = source["views"]
    nodes = [(vi, r, int(mu.area({**r,"counts":r["counts"].encode()})))
             for vi,view in enumerate(views) for r in view["rles"]]
    groups = []
    for vi,rle,area in sorted(nodes,key=lambda x:-x[2]):
        available = [i for i,g in enumerate(groups) if g["rles"][vi] is None]
        match, score = -1, 0.
        if available:
            a = [{**rle,"counts":rle["counts"].encode()}]
            refs = [{**groups[i]["reference"],"counts":groups[i]["reference"]["counts"].encode()} for i in available]
            overlap = np.asarray(mu.iou(a,refs,[0]*len(refs)))[0]
            j = int(overlap.argmax())
            if overlap[j]>=.25:
                match, score = available[j], overlap[j]
        if match<0:
            groups.append(dict(rles=[None]*len(views), reference=rle))
            match = len(groups)-1
        groups[match]["rles"][vi] = rle
    records = []
    for group in groups:
        boxes = [mu.toBbox({**r,"counts":r["counts"].encode()}) for r in group["rles"] if r]
        boxes = np.asarray(boxes)
        bounds = [boxes[:,0].min(),boxes[:,1].min(),(boxes[:,0]+boxes[:,2]).max(),(boxes[:,1]+boxes[:,3]).max()]
        roi=square_roi(bounds,cfg)
        rotation=0;flip=False
        if part=="train":
            rng=np.random.default_rng(cfg["seed"]+int(b.digest([source["stem"],len(records)])[:8],16))
            variant={**cfg,"roi_context":float(rng.uniform(1.4,2.0))}
            roi=square_roi(bounds,variant);side=roi[2]-roi[0]
            for axis in (0,1):
                lower=max(0,int(math.ceil(bounds[axis+2]-side)))
                upper=min(2048-side,int(math.floor(bounds[axis])))
                if lower<=upper:
                    roi[axis]=int(np.clip(round(roi[axis]+rng.uniform(-.12,.12)*side),lower,upper))
                    roi[axis+2]=roi[axis]+side
            rotation=int(rng.integers(4));flip=bool(rng.random()<.5)
        records.append(dict(stem=source["stem"], part=part, roi=roi,rotation=rotation,flip=flip,
            rles=group["rles"], nviews=len(views), positive=True, image=source["cache"]))
    # Dark, on-disk negatives with no annotation in ANY available annotator view.
    image = cv2.imread(source["cache"])
    union = np.zeros((2048,2048),bool)
    for _,r,_ in nodes:
        union |= b.decode(r)
    rng = np.random.default_rng(cfg["seed"]+int(b.digest(source["stem"])[:8],16))
    points = rng.integers(200,1848,size=(160,2))
    points = points[((points-1024)**2).sum(1)<820**2]
    intensity = image[points[:,0],points[:,1],1]
    negative_boxes = []
    for j in np.argsort(intensity):
        y,x = points[j]
        if intensity[j]<10:
            continue
        roi = square_roi([x-32,y-32,x+32,y+32],cfg)
        x0,y0,x1,y1 = roi
        if union[y0:y1,x0:x1].any():
            continue
        if any(abs(x0-a[0])+abs(y0-a[1])<150 for a in negative_boxes):
            continue
        negative_boxes.append(roi)
        records.append(dict(stem=source["stem"],part=part,roi=roi,rles=[None]*len(views),
            nviews=len(views),positive=False,image=source["cache"],
            rotation=int(rng.integers(4)) if part=="train" else 0,
            flip=bool(rng.random()<.5) if part=="train" else False))
        if len(negative_boxes)==2:
            break
    return records


def crop_transform(array,record):
    transformed=np.rot90(array,int(record.get("rotation",0)),axes=(-2,-1))
    if record.get("flip",False):transformed=transformed[...,::-1]
    return np.ascontiguousarray(transformed)


def prepare_sam_bank(cfg, manifest):
    run = Path(cfg["run"])
    descriptor = run/"bank.json"
    if descriptor.is_file():
        return b.load_json(descriptor)
    rows = [(s,"train") for s in manifest["splits"]["train"]]
    rows += [(s,"validation") for s in manifest["splits"]["calibration"]]
    groups = list(b.bounded_map(lambda pair:grouped_records(pair[0],pair[1],cfg), rows,4))
    train = [r for group in groups for r in group if r["part"]=="train"]
    valid = [r for group in groups for r in group if r["part"]=="validation"]
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(train);rng.shuffle(valid)
    if cfg["mode"]=="smoke":
        train = train[:8]
    records = valid[:cfg["validation_records"]]+train
    n = len(records)
    v = max(r["nviews"] for r in records)
    feature_bytes=2 if cfg["feature_storage"]=="float16" else 1
    needed = n*(256*64*64*feature_bytes+v*256*256+256*256+256*4)+1_000_000_000
    if shutil.disk_usage(run).free < needed+2*2**30:
        raise OSError(f"Embedding/target bank needs {needed/2**30:.1f} GiB plus reserve")
    bank = dict(records=records,count=n,max_views=v, directory=str(run/"bank"))
    folder = Path(bank["directory"]);folder.mkdir(exist_ok=True)
    feature_dtype=np.float16 if cfg["feature_storage"]=="float16" else np.int8
    arrays = dict(embedding=(feature_dtype,(n,256,64,64)), scale=(np.float32,(n,256)),
        targets=(np.uint8,(n,v,256,256)),seed=(np.uint8,(n,256,256)),done=(np.uint8,(n,)))
    for name,(dtype,shape) in arrays.items():
        value = np.lib.format.open_memmap(folder/(name+".npy"),mode="w+",dtype=dtype,shape=shape)
        if name in ("targets","seed","done"):
            value[:] = 0
        value.flush();del value
    targets = np.load(folder/"targets.npy",mmap_mode="r+")
    seeds = np.load(folder/"seed.npy",mmap_mode="r+")
    def target_record(pair):
        index,record = pair
        x0,y0,x1,y1 = record["roi"]
        result = np.zeros((v,256,256),np.uint8)
        for j,rle in enumerate(record["rles"]):
            if rle:
                local = b.decode(rle)[y0:y1,x0:x1].astype(np.float32)
                resized = cv2.resize(local,(256,256),interpolation=cv2.INTER_AREA if x1-x0>256 else cv2.INTER_LINEAR)
                result[j] = np.rint(resized*255).astype(np.uint8)
        if record["positive"]:
            seed = result.max(0)
        else:
            image = cv2.imread(record["image"])[y0:y1,x0:x1,1]
            gray = cv2.resize(image,(256,256),interpolation=cv2.INTER_LINEAR)
            threshold = min(float(np.percentile(gray,12)),float(gray.mean()-gray.std()*.5))
            binary = (gray<=threshold).astype(np.uint8)
            pieces = list(b.components(binary.astype(np.float32),.5,3,64))
            seed = np.zeros((256,256),np.uint8)
            if pieces:
                box,mask = max(pieces,key=lambda value:int(value[1].sum()))
                a,c,d,e = box;seed[c:e,a:d] = mask.astype(np.uint8)*200
            else:
                cv2.circle(seed,(128,128),5,200,-1)
        return index,crop_transform(result,record),crop_transform(seed,record)
    for i,target,seed in b.bounded_map(target_record,enumerate(records),4):
        targets[i],seeds[i] = target,seed
    targets.flush();seeds.flush()
    b.save_json(descriptor,bank)
    print(f"Prepared {n} cropped training/validation records; image encoder stays frozen",flush=True)
    return bank


def quantize_embedding(embedding):
    b.require_finite(embedding,"SAM image embedding")
    scale = embedding.abs().amax(dim=(-2,-1),keepdim=True).clamp_min(1e-6)/127
    quantized = torch.round(embedding/scale).clamp(-127,127).to(torch.int8)
    return quantized, scale.squeeze(-1).squeeze(-1)


def stored_embedding(embedding,storage):
    if storage=="int8":return quantize_embedding(embedding)
    if storage!="float16":raise ValueError("Unsupported feature storage")
    b.require_finite(embedding,"SAM encoder output before storage")
    stored=embedding.to(torch.float16)
    b.require_finite(stored,"SAM feature storage conversion")
    scale=torch.ones(embedding.shape[:2],device=embedding.device,dtype=torch.float32)
    return stored,scale


class CropImageDataset(Dataset):
    def __init__(self, records, indices):
        self.records,self.indices = records,indices
        self.last = (None,None)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        identity = int(self.indices[index]);record = self.records[identity]
        if self.last[0]!=record["image"]:
            self.last = record["image"],cv2.imread(record["image"])
        x0,y0,x1,y1 = record["roi"]
        # The original pipeline's global standardized channel, replicated as RGB.
        gray = self.last[1][y0:y1,x0:x1,0]
        image = cv2.resize(gray,(1024,1024),interpolation=cv2.INTER_LINEAR)
        image = crop_transform(image,record)
        image = np.repeat(image[None],3,axis=0)
        return torch.from_numpy(np.ascontiguousarray(image)),identity


def make_sam(cfg):
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=cfg["sam_checkpoint"])
    sam.eval().requires_grad_(False)
    return sam


def cache_sam_embeddings(config_path, slot):
    cfg = b.load_json(config_path);bank = b.load_json(Path(cfg["run"])/"bank.json")
    device = torch.device(f"cuda:{cfg['gpu_ids'][slot]}");torch.cuda.set_device(device)
    sam = make_sam(cfg).to(device)
    folder = Path(bank["directory"])
    done = np.load(folder/"done.npy",mmap_mode="r+")
    embedding = np.load(folder/"embedding.npy",mmap_mode="r+")
    scale = np.load(folder/"scale.npy",mmap_mode="r+")
    indices = [i for i in range(slot,bank["count"],len(cfg["gpu_ids"])) if not done[i]]
    dataset = CropImageDataset(bank["records"],indices)
    workers = cfg["cpu_workers"]//len(cfg["gpu_ids"])
    options = dict(num_workers=workers,pin_memory=True)
    if workers:options["prefetch_factor"]=2
    loader = DataLoader(dataset,batch_size=cfg["encode_batch"],**options)
    timing=[];encoded=0
    for images,ids in loader:
        sam_remaining(cfg,180)
        if cfg["mode"]!="smoke" and time.time()>cfg["session_started"]+cfg["cache_hours"]*3600:
            break
        images = images.to(device,dtype=torch.float32,non_blocking=True)
        torch.cuda.synchronize(device);before=time.monotonic()
        with torch.inference_mode():
            features = sam.image_encoder((images-sam.pixel_mean)/sam.pixel_std)
            q,s = stored_embedding(features,cfg["feature_storage"])
        torch.cuda.synchronize(device)
        timing.append((time.monotonic()-before)/len(ids))
        for identity,value,factor in zip(ids.tolist(),q.cpu().numpy(),s.cpu().numpy()):
            embedding[identity],scale[identity],done[identity] = value,factor,1
        encoded += len(ids)
        if encoded%40==0:
            print(f"SAM cache GPU {device.index}: {encoded}/{len(indices)} crops",flush=True)
    embedding.flush();scale.flush();done.flush()
    result=dict(gpu=str(device),gpu_name=torch.cuda.get_device_name(device),encoded=encoded,
        seconds_per_crop=float(np.median(timing[1:] or timing)) if timing else None,
        peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30,precision="float32")
    b.save_json(Path(cfg["run"])/f"cache_gpu_{slot}.json",result)
    print(json.dumps(result),flush=True)


class AlignedSAMDecoder(nn.Module):
    """One prompt set per image, using SAM's original transformer and heads.

    SAM's public decoder expands one image over many prompts; here images and
    prompt sets already share a batch dimension. Architecture/weights are kept.
    SAM is Copyright Meta Platforms, Inc.; upstream license Apache-2.0:
    https://github.com/facebookresearch/segment-anything
    """
    def __init__(self,decoder):
        super().__init__();self.decoder=decoder
        decoder.requires_grad_(True)
        for head in decoder.output_hypernetworks_mlps[1:]:head.requires_grad_(False)
        # Adapt frozen natural-image features to solar texture without re-encoding
        # them for every training epoch. Zero initialization preserves SAM exactly.
        self.adapter=nn.Sequential(nn.Conv2d(256,64,3,padding=1),nn.GELU(),nn.Conv2d(64,256,1))
        nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
    def forward(self,image,sparse,dense,position):
        decoder=self.decoder;batch,channels,height,width=image.shape
        image=image+self.adapter(image)
        tokens=torch.cat([decoder.iou_token.weight,decoder.mask_tokens.weight],0)
        tokens=torch.cat([tokens.unsqueeze(0).expand(batch,-1,-1),sparse],1)
        hidden,updated=decoder.transformer(image+dense,position.expand(batch,-1,-1,-1),tokens)
        feature=decoder.output_upscaling(updated.transpose(1,2).reshape(batch,channels,height,width))
        coefficient=decoder.output_hypernetworks_mlps[0](hidden[:,1])
        mask=(coefficient[:,None]@feature.flatten(2)).reshape(batch,1,feature.shape[-2],feature.shape[-1])
        quality=decoder.iou_prediction_head(hidden[:,0])[:,0]
        return mask,quality


def prompt_tensors(seed, rng=None):
    seed=np.asarray(seed,np.float32)
    binary=(seed>max(.03,min(.25,float(seed.max())*.5))).astype(np.uint8)
    if not binary.any():
        y,x=np.unravel_index(int(seed.argmax()),seed.shape);binary[y,x]=1
    distance=cv2.distanceTransform(binary,cv2.DIST_L2,3)
    y,x=np.unravel_index(int(distance.argmax()),distance.shape)
    yy,xx=np.nonzero(binary)
    pad=8 if rng is None else int(rng.integers(4,20))
    box=np.array([max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),
        min(256,int(xx.max())+pad+1),min(256,int(yy.max())+pad+1)],np.float32)*4
    point=np.array([[(x+.5)*4,(y+.5)*4]],np.float32)
    clipped=np.clip(seed,.02,.98)
    mask=np.log(clipped/(1-clipped)).astype(np.float32)[None]
    return mask,point,box


class EmbeddingDataset(Dataset):
    def __init__(self,bank,indices,seed,augment):
        self.bank,self.indices,self.seed,self.augment,self.epoch=bank,list(indices),seed,augment,0
        self.arrays=None
    def __len__(self):return len(self.indices)
    def __getitem__(self,index):
        if self.arrays is None:
            folder=Path(self.bank["directory"])
            self.arrays={k:np.load(folder/(k+".npy"),mmap_mode="r") for k in ("embedding","scale","targets","seed")}
        identity=self.indices[index];record=self.bank["records"][identity]
        rng=np.random.default_rng(self.seed+identity+self.epoch*10_000_019)
        targets=np.array(self.arrays["targets"][identity],copy=True)
        nonempty=np.flatnonzero(targets.sum((1,2))>0)
        target=targets[int(rng.choice(nonempty))] if len(nonempty) else np.zeros((256,256),np.uint8)
        seed=np.array(self.arrays["seed"][identity],np.float32)/255
        if self.augment:
            operation=rng.random()
            if operation<.25:
                seed=cv2.erode(seed,np.ones((3,3),np.uint8))
            elif operation<.5:
                seed=cv2.dilate(seed,np.ones((3,3),np.uint8))
            if record["positive"] and rng.random()<.25:
                if rng.random()<.5:seed[:int(rng.integers(64,128))]=0
                else:seed[:,int(rng.integers(128,192)):]=0
            if seed.max()<.05:seed=target.astype(np.float32)/255
            seed=cv2.GaussianBlur(seed,(5,5),rng.uniform(.5,1.4))*rng.uniform(.45,.95)
        mask,point,box=prompt_tensors(seed,rng if self.augment else None)
        return dict(embedding=torch.from_numpy(np.array(self.arrays["embedding"][identity],copy=True)),
            scale=torch.from_numpy(np.array(self.arrays["scale"][identity],copy=True)),
            target=torch.from_numpy(target[None].copy()),views=torch.from_numpy(targets),
            nviews=torch.tensor(record["nviews"]),mask=torch.from_numpy(mask),
            point=torch.from_numpy(point),box=torch.from_numpy(box),
            box_mode=torch.tensor(rng.random()<.5 if self.augment else identity%2==0))


def sam_prompts(prompt, batch, device):
    with torch.no_grad():
        points=batch["point"].to(device)
        masks=batch["mask"].to(device)
        sparse,dense=prompt(points=(points,torch.ones(points.shape[:2],device=device,dtype=torch.int64)),
                            boxes=None,masks=masks)
        box_sparse,_=prompt(points=None,boxes=batch["box"].to(device),masks=None)
        choose=batch["box_mode"].to(device)[:,None,None]
        sparse=torch.where(choose,box_sparse,sparse)
        position=prompt.get_dense_pe()
    image=batch["embedding"].to(device).float()*batch["scale"].to(device)[:,:,None,None]
    return image,sparse,dense,position


def soft_skeleton(mask,iterations=3):
    def erode(x):
        return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
    def opened(x):return F.max_pool2d(erode(x),3,1,1)
    result=F.relu(mask-opened(mask))
    for _ in range(iterations):
        mask=erode(mask);delta=F.relu(mask-opened(mask));result=result+F.relu(delta-result*delta)
    return result


def refiner_loss(logits,quality,batch,device):
    with torch.autocast(device_type=device.type,enabled=False):
        logits,quality=logits.float(),quality.float()
        target=batch["target"].to(device).float()/255
        probability=logits.sigmoid();axes=(1,2,3)
        bce=F.binary_cross_entropy_with_logits(logits,target)
        dice=(1-(2*(probability*target).sum(axes)+1)/(probability.sum(axes)+target.sum(axes)+1)).mean()
        predicted_line=soft_skeleton(probability);true_line=soft_skeleton(target)
        precision=((predicted_line*target).sum(axes)+1)/(predicted_line.sum(axes)+1)
        sensitivity=((true_line*probability).sum(axes)+1)/(true_line.sum(axes)+1)
        cldice=(1-(2*precision*sensitivity+1e-6)/(precision+sensitivity+1e-6)).mean()
        with torch.no_grad():
            predicted=(probability[:,0]>.5).float()[:,None]
            views=batch["views"].to(device).float()/255
            intersection=(predicted*views).sum((-2,-1))
            union=predicted.sum((-2,-1))+views.sum((-2,-1))-intersection
            overlap=intersection/union.clamp_min(1)
            active=torch.arange(views.shape[1],device=device)[None]<batch["nviews"].to(device)[:,None]
            credit=(overlap*(overlap>.5)*active).sum(1)/batch["nviews"].to(device).clamp_min(1)
        qloss=F.smooth_l1_loss(quality,credit,beta=.1)
        return .45*bce+.40*dice+.10*cldice+.05*qloss


def launch_sam(cfg,task,extra=None,distributed=False):
    command=[sys.executable]
    if distributed and len(cfg["gpu_ids"])>1:
        command += ["-m","torch.distributed.run","--standalone","--nnodes=1",f"--nproc_per_node={len(cfg['gpu_ids'])}"]
    command += [str(SAM_WORKER),"--config",str(Path(cfg["run"])/"config.json"),"--task",task]
    command += list(map(str,extra or []))
    print("Launching:"," ".join(command),flush=True)
    return subprocess.Popen(command,start_new_session=os.name!="nt")


def stop_sam_process(process):
    if process.poll() is not None:return
    if os.name=="nt":
        subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(process.pid),signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name=="nt":process.kill()
        else:
            import signal
            os.killpg(os.getpgid(process.pid),signal.SIGKILL)


def await_sam(cfg,processes):
    try:
        for process in processes:
            code=process.wait(timeout=max(1,sam_remaining(cfg,30)))
            if code:raise RuntimeError(f"SAM worker failed with exit status {code}")
    finally:
        for process in processes:stop_sam_process(process)


def encode_sam_bank(cfg):
    processes=[launch_sam(cfg,"cache",["--slot",i]) for i in range(len(cfg["gpu_ids"]))]
    await_sam(cfg,processes)
    bank=b.load_json(Path(cfg["run"])/"bank.json")
    done=np.load(Path(bank["directory"])/"done.npy",mmap_mode="r")
    train=[i for i,r in enumerate(bank["records"]) if r["part"]=="train" and done[i]]
    valid=[i for i,r in enumerate(bank["records"]) if r["part"]=="validation" and done[i]]
    if not train or not valid or (cfg["mode"]=="full" and sum(bank["records"][i]["positive"] for i in train)<1000):
        raise RuntimeError("Too few encoded train/validation records within the cache time budget")
    b.save_json(Path(cfg["run"])/"bank_ready.json",dict(train_indices=train,validation_indices=valid,
        encoded_train=len(train),encoded_validation=len(valid),total_records=bank["count"]))
    return train,valid


def train_sam_decoder(config_path):
    cfg=b.load_json(config_path);run=Path(cfg["run"])
    device,rank,world=b.device_setup(cfg,distributed=True)
    b.seed_all(cfg["seed"])
    if (run/"decoder_complete.json").is_file():return
    bank=b.load_json(run/"bank.json");ready=b.load_json(run/"bank_ready.json")
    training=EmbeddingDataset(bank,ready["train_indices"],cfg["seed"],True)
    validation=EmbeddingDataset(bank,ready["validation_indices"][rank::world],cfg["seed"],False)
    sampler=DistributedSampler(training,world,rank,shuffle=True,seed=cfg["seed"])
    workers=cfg["cpu_workers"]//world
    options=dict(num_workers=workers,pin_memory=True,persistent_workers=False)
    if workers:options["prefetch_factor"]=2
    loader=DataLoader(training,batch_size=cfg["batch_per_gpu"],sampler=sampler,drop_last=False,**options)
    valid_loader=DataLoader(validation,batch_size=cfg["batch_per_gpu"],num_workers=0)
    steps=min(cfg["steps"],len(loader))
    sam=make_sam(cfg)
    prompt=sam.prompt_encoder.to(device).eval()
    model=AlignedSAMDecoder(sam.mask_decoder).to(device)
    wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False) if world>1 else model
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    updates=0;best=float("inf");begin=0
    last=run/"decoder_training.pt"
    if last.is_file():
        state=torch.load(last,map_location="cpu",weights_only=False);b.require_finite_state(state)
        if state["fingerprint"]!=cfg["fingerprint"]:raise ValueError("SAM resume fingerprint changed")
        model.load_state_dict(state["model"]);ema.load_state_dict(state["ema"]);optimizer.load_state_dict(state["optimizer"])
        begin,updates,best=state["epoch"],state["updates"],state["best"]
    before=time.time()
    stop_at=min(before+cfg["train_hours"]*3600,cfg["session_started"]+(cfg["total_hours"]-.9)*3600)
    resource=dict(rank=rank,world_size=world,gpu_name=torch.cuda.get_device_name(device),
        batch_per_gpu=cfg["batch_per_gpu"],loader_workers=workers,steps=steps,
        train_records=len(training),validation_records=len(validation),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_image_encoder=True,frozen_prompt_encoder=True,precision="float32")
    b.save_json(run/f"decoder_resources_rank_{rank}.json",resource)
    print("SAM decoder training:",json.dumps(resource),flush=True)
    completed=begin
    for epoch in range(begin,cfg["epochs"]):
        started=time.time();training.epoch=epoch;sampler.set_epoch(epoch);model.train()
        total=0.;torch.cuda.reset_peak_memory_stats(device)
        for step,batch in enumerate(loader):
            if step>=steps:break
            progress=max((epoch+step/max(steps,1))/cfg["epochs"],min(1.,(time.time()-before)/max(stop_at-before,1)))
            warm=min(1.,(epoch*steps+step+1)/max(steps,1))
            lr=cfg["lr"]*warm*(.03+.97*(1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups:group["lr"]=lr
            optimizer.zero_grad(set_to_none=True)
            image,sparse,dense,position=sam_prompts(prompt,batch,device)
            logits,quality=wrapped(image,sparse,dense,position)
            loss=refiner_loss(logits,quality,batch,device)
            b.require_finite(loss,"SAM decoder loss",world>1)
            loss.backward()
            norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=False)
            b.require_finite(norm,"SAM decoder gradient norm",world>1)
            optimizer.step();updates+=1
            b.update_ema(ema,model,min(cfg["ema_decay"],(1+updates)/(10+updates)))
            total+=float(loss.detach())
            if (step+1)%40==0:print(f"SAM rank {rank}: epoch {epoch+1}/{cfg['epochs']}, step {step+1}/{steps}, loss {total/(step+1):.4f}",flush=True)
        val_sum=0.;val_count=0
        evaluate=epoch==0 or (epoch+1)%cfg["validation_every"]==0 or epoch+1==cfg["epochs"]
        if evaluate:
            with torch.inference_mode():
                for batch in valid_loader:
                    outputs=ema(*sam_prompts(prompt,batch,device))
                    loss=refiner_loss(*outputs,batch,device)
                    b.require_finite(loss,"SAM validation loss",world>1)
                    number=len(batch["nviews"]);val_sum+=float(loss)*number;val_count+=number
        stats=torch.tensor([total/max(steps,1),val_sum,val_count],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(stats)
        stats=stats.cpu().numpy();val=float(stats[1]/max(stats[2],1)) if evaluate else None
        improved=evaluate and val<best
        if improved:best=val
        completed=epoch+1
        if rank==0:
            checkpoint=dict(state=ema.state_dict(),epoch=completed,validation_loss=val,
                fingerprint=cfg["fingerprint"],sam_sha256=cfg["sam_sha256"],encoder_frozen=True)
            if improved:b.checkpoint_save(run/"sam_decoder_best.pt",checkpoint)
            b.checkpoint_save(run/"sam_decoder_last.pt",checkpoint)
            b.checkpoint_save(last,dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                epoch=completed,updates=updates,best=best,fingerprint=cfg["fingerprint"]))
            row=dict(epoch=completed,training_loss=float(stats[0]/world),validation_loss=val,
                seconds=time.time()-started,updates=updates,lr=lr,peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30)
            with (run/"decoder_history.jsonl").open("a",encoding="utf8") as stream:stream.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        stop=torch.tensor(int(time.time()+max(30.,time.time()-started)*1.2>stop_at),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item():break
    if rank==0:
        b.save_json(run/"decoder_complete.json",dict(epochs=completed,best_validation_loss=best,
            checkpoint_sha256=b.file_sha(run/"sam_decoder_best.pt"),seconds=time.time()-before))
        last.unlink(missing_ok=True)


def train_refiner(cfg):
    await_sam(cfg,[launch_sam(cfg,"train",distributed=True)])
    existing=Path(cfg["run"])/"inference_plan.json"
    if existing.is_file():
        cfg["max_proposals"]=b.load_json(existing)["max_proposals_per_image"]
        b.save_json(Path(cfg["run"])/"config.json",cfg)
        return cfg
    timings=[b.load_json(Path(cfg["run"])/f"cache_gpu_{i}.json") for i in range(len(cfg["gpu_ids"]))]
    cost=max(x["seconds_per_crop"] or 1. for x in timings)
    manifest=b.load_json(Path(cfg["run"])/"manifest.json")
    count=sum(len(manifest["splits"][part]) for part in ("calibration","report","test"))
    cap=int(max(1,sam_remaining(cfg)-900)*.65*len(cfg["gpu_ids"])/(max(cost,.01)*max(count,1)))
    cfg["max_proposals"]=min(cfg["max_proposals"],max(2,cap))
    b.save_json(Path(cfg["run"])/"config.json",cfg)
    b.save_json(Path(cfg["run"])/"inference_plan.json",dict(seconds_per_crop=cost,
        max_proposals_per_image=cfg["max_proposals"],images=count,remaining_seconds=sam_remaining(cfg)))
    return cfg


def prior_quality(cfg):
    return load_quality(Path(cfg["prior_root"])/"quality_model.joblib")


def load_quality(path):
    # Some sklearn/Cython wheels serialize their extension under the short name
    # '_loss'. Resolve that name to the SAME installed sklearn extension; do not
    # change the estimator or refit it during inference.
    import importlib
    import joblib
    sys.modules.setdefault("_loss",importlib.import_module("sklearn._loss._loss"))
    return joblib.load(path)


def prior_instances(source,manifest,quality):
    recipe=manifest["prior_calibration"]["recipe"]
    candidates=b.load_json(source["prior_candidates"])[recipe["set"]]
    if recipe["quality"]:
        scores=(source["prior_oof"] if "prior_oof" in source else
                np.clip(quality.predict(np.asarray([c["features"] for c in candidates],np.float32)),0,1) if candidates else [])
    else:scores=[c["features"][1] for c in candidates]
    if len(scores)!=len(candidates):raise ValueError("Prior OOF candidate alignment mismatch")
    selected=[copy.deepcopy(c) for c,s in zip(candidates,scores) if s>=recipe["cutoff"] and c["area"]>=recipe["min_area"]]
    return selected


def propose_seeds(source,cfg,manifest,quality,probability,image):
    base=prior_instances(source,manifest,quality)
    candidates=[]
    for c in base:c["priority"]=2.;candidates.append(c)
    for threshold in cfg["proposal_thresholds"]:
        for box,mask in b.components(probability[0],threshold,cfg["minimum_seed_area"],192):
            if max(box[2]-box[0],box[3]-box[1])>1024:continue
            c=b.make_candidate(box,mask,probability[0],probability[1],image)
            c["priority"]=float(c["features"][1])
            candidates.append(c)
    chosen=[];encoded=[]
    for c in sorted(candidates,key=lambda x:-x["priority"]):
        current={**c["rle"],"counts":c["rle"]["counts"].encode()}
        if encoded and np.asarray(mu.iou([current],encoded,[0]*len(encoded))).max()>.80:continue
        chosen.append(c);encoded.append(current)
        if len(chosen)>=cfg["max_proposals"]:break
    return base,chosen


def put_seed_in_roi(candidate,roi,probability):
    x0,y0,x1,y1=roi;a,c,d,e=candidate["box"]
    result=np.zeros((y1-y0,x1-x0),np.float32)
    left,top,right,bottom=max(x0,a),max(y0,c),min(x1,d),min(y1,e)
    if right>left and bottom>top:
        local=b.decode(candidate["local_rle"])
        result[top-y0:bottom-y0,left-x0:right-x0]=local[top-c:bottom-c,left-a:right-a]*probability[top:bottom,left:right]
    return result


def infer_sam_worker(config_path,part,slot):
    cfg=b.load_json(config_path);run=Path(cfg["run"]);manifest=b.load_json(run/"manifest.json")
    device=torch.device(f"cuda:{cfg['gpu_ids'][slot]}");torch.cuda.set_device(device)
    sam=make_sam(cfg).to(device)
    model=AlignedSAMDecoder(sam.mask_decoder).to(device)
    checkpoint=torch.load(run/"sam_decoder_best.pt",map_location="cpu",weights_only=False)
    b.require_finite_state(checkpoint)
    if checkpoint["fingerprint"]!=cfg["fingerprint"]:raise ValueError("Wrong SAM decoder checkpoint")
    model.load_state_dict(checkpoint["state"]);model.eval().requires_grad_(False)
    quality=prior_quality(cfg)
    output=run/"candidates"/part;output.mkdir(parents=True,exist_ok=True)
    sources=manifest["splits"][part][slot::len(cfg["gpu_ids"])]
    for index,source in enumerate(sources):
        sam_remaining(cfg,60)
        target=output/(source["stem"]+".json")
        if target.is_file():continue
        image=cv2.imread(source["cache"])
        with np.load(source["prior_map"]) as values:probability=values["prob"].astype(np.float32)/255
        base,seeds=propose_seeds(source,cfg,manifest,quality,probability,image)
        for c in base:c["features"]+= [0.,0.,1.,0.]
        sets={"baseline":base}
        for threshold in cfg["mask_thresholds"]:sets[f"refined_{threshold}"]=[]
        for start in range(0,len(seeds),cfg["encode_batch"]):
            seed_batch=seeds[start:start+cfg["encode_batch"]]
            rois=[square_roi(c["box"],cfg) for c in seed_batch]
            crops=[];masks=[];points=[];boxes=[];seed_images=[]
            for candidate,roi in zip(seed_batch,rois):
                x0,y0,x1,y1=roi
                gray=cv2.resize(image[y0:y1,x0:x1,0],(1024,1024),interpolation=cv2.INTER_LINEAR)
                crops.append(np.repeat(gray[None],3,axis=0))
                seed=put_seed_in_roi(candidate,roi,probability[0]);seed_images.append(seed)
                seed=cv2.resize(seed,(256,256),interpolation=cv2.INTER_AREA)
                mask,point,box=prompt_tensors(seed);masks.append(mask);points.append(point);boxes.append(box)
            pixels=torch.from_numpy(np.stack(crops)).to(device,dtype=torch.float32)
            with torch.inference_mode():
                features=sam.image_encoder((pixels-sam.pixel_mean)/sam.pixel_std)
                q,s=stored_embedding(features,cfg["feature_storage"]);features=q.float()*s[:,:,None,None]
                mask_inputs=torch.from_numpy(np.stack(masks)).to(device)
                point_inputs=torch.from_numpy(np.stack(points)).to(device)
                box_inputs=torch.from_numpy(np.stack(boxes)).to(device)
                sparse_point,dense=sam.prompt_encoder(points=(point_inputs,torch.ones(point_inputs.shape[:2],device=device,dtype=torch.int64)),boxes=None,masks=mask_inputs)
                sparse_box,_=sam.prompt_encoder(points=None,boxes=box_inputs,masks=None)
                for prompt_kind,sparse in (("point",sparse_point),("box",sparse_box)):
                    logits,scores=model(features,sparse,dense,sam.prompt_encoder.get_dense_pe())
                    b.require_finite(logits,"SAM inference mask");b.require_finite(scores,"SAM inference quality")
                    probabilities=logits.sigmoid().cpu().numpy()[:,0];scores=scores.cpu().numpy()
                    for candidate,roi,seed,predicted,score in zip(seed_batch,rois,seed_images,probabilities,scores):
                        x0,y0,x1,y1=roi
                        local=cv2.resize(predicted,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
                        for threshold in cfg["mask_thresholds"]:
                            pieces=list(b.components(local,threshold,20,96))
                            if not pieces:continue
                            ranked=[]
                            for box,piece in pieces:
                                a,c,d,e=box;support=float(seed[c:e,a:d][piece].sum())
                                if support>0:ranked.append((support,box,piece))
                            if not ranked:continue
                            _,box,piece=max(ranked,key=lambda x:x[0])
                            a,c,d,e=box;global_box=[x0+a,y0+c,x0+d,y0+e]
                            value=b.make_candidate(global_box,piece,probability[0],probability[1],image)
                            coverage=float(seed[c:e,a:d][piece].sum()/max(seed.sum(),1e-6))
                            value["features"] += [float(np.clip(score,0,1)),float(local[c:e,a:d][piece].mean()),coverage,1.]
                            value["prompt_kind"]=prompt_kind
                            sets[f"refined_{threshold}"].append(value)
        if part=="calibration":
            for candidates in sets.values():
                for c,y in zip(candidates,b.candidate_targets(source,candidates)):c["target"]=float(y)
        b.save_json(target,sets)
        print(f"SAM GPU {device.index}: {part} {index+1}/{len(sources)}, {len(seeds)} crop encodings",flush=True)


def predict_refiner(cfg,part):
    processes=[launch_sam(cfg,"infer",["--slot",i,"--part",part]) for i in range(len(cfg["gpu_ids"]))]
    await_sam(cfg,processes)
    folder=Path(cfg["run"])/"candidates"/part
    sources=b.load_json(Path(cfg["run"])/"manifest.json")["splits"][part]
    if {p.stem for p in folder.glob("*.json")}!={s["stem"] for s in sources}:
        raise RuntimeError("Incomplete SAM candidate partition")
    return folder


def fit_refiner_quality(cfg,folder):
    import joblib
    manifest=b.load_json(Path(cfg["run"])/"manifest.json")
    data={};x=[];y=[];dates=[];index=[]
    for source in manifest["splits"]["calibration"]:
        sets=b.load_json(Path(folder)/(source["stem"]+".json"));data[source["stem"]]=sets
        for name,candidates in sets.items():
            for j,c in enumerate(candidates):
                x.append(c["features"]);y.append(c["target"]);dates.append(source["date"]);index.append((source["stem"],name,j))
    if not x:raise RuntimeError("No SAM calibration candidates")
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);dates=np.asarray(dates)
    unique=np.array(sorted(set(dates)));np.random.default_rng(cfg["seed"]).shuffle(unique)
    folds=min(cfg["quality_folds"],len(unique))
    if folds<2:raise ValueError("Quality crossfitting requires two calibration dates")
    assignment={date:i%folds for i,date in enumerate(unique)};oof=np.zeros(len(y),np.float32)
    for fold in range(folds):
        valid=np.asarray([assignment[d]==fold for d in dates])
        model=b.quality_estimator(cfg["seed"]+fold)
        model.fit(x[~valid],y[~valid]);oof[valid]=np.clip(model.predict(x[valid]),0,1)
    for (stem,name,j),score in zip(index,oof):data[stem][name][j]["quality_oof"]=float(score)
    model=b.quality_estimator(cfg["seed"]);model.fit(x,y)
    joblib.dump(model,Path(cfg["run"])/"refiner_quality.joblib")
    b.save_json(Path(cfg["run"])/"quality_audit.json",dict(features=list(b.FEATURES)+EXTRA_FEATURES,
        folds=folds,group="calibration date",rows=len(x),dates=len(unique),report_used=False,
        target="Expected credited IoU over original annotation views",crossfit_mae=float(np.abs(oof-y).mean())))
    return model,data


def refined_family(sets,recipe):
    if recipe["family"]=="prior":return sets["baseline"]
    refined=sets[f"refined_{recipe['threshold']}"]
    return refined+sets["baseline"] if recipe["family"]=="fusion" else refined


def select_refiner_instances(sets,recipe,quality=None,oof=False):
    candidates=refined_family(sets,recipe)
    if recipe["family"]=="prior":return [c["rle"] for c in candidates]
    if not candidates:return []
    scores=np.asarray([c["quality_oof"] for c in candidates]) if oof else np.clip(
        quality.predict(np.asarray([c["features"] for c in candidates],np.float32)),0,1)
    owned=np.zeros((2048,2048),bool);accepted=[];rles=[];areas=[]
    for i in np.argsort(-scores,kind="stable"):
        c=candidates[i]
        if scores[i]<recipe["cutoff"] or c["area"]<recipe["min_area"]:continue
        encoded={**c["rle"],"counts":c["rle"]["counts"].encode()}
        if rles:
            overlap=np.asarray(mu.iou([encoded],rles,[0]*len(rles)))[0]
            intersection=overlap*(c["area"]+np.asarray(areas))/(1+overlap)
            if overlap.max()>.5 or (intersection/np.minimum(c["area"],areas)).max()>.70:continue
        x0,y0,x1,y1=c["box"]
        local=c.get("_decoded")
        if local is None:local=b.decode(c["local_rle"])
        mask=local & ~owned[y0:y1,x0:x1]
        if int(mask.sum())<recipe["min_area"]:continue
        owned[y0:y1,x0:x1]|=mask
        full=b.full_rle(c["box"],mask)
        accepted.append(full);rles.append({**full,"counts":full["counts"].encode()});areas.append(int(mask.sum()))
    return accepted


def calibrate_refiner(cfg,folder):
    sam_remaining(cfg,600)
    model,data=fit_refiner_quality(cfg,folder)
    manifest=b.load_json(Path(cfg["run"])/"manifest.json")
    recipes=[dict(family=family,threshold=t,cutoff=cutoff,min_area=area)
        for family in ("refined","fusion") for t in cfg["mask_thresholds"]
        for cutoff in (.15,.20,.25,.30,.35) for area in (40,100,200)]
    def score_source(source):
        sets=data[source["stem"]]
        for candidates in sets.values():
            for c in candidates:c["_decoded"]=b.decode(c["local_rle"])
        result=np.zeros((len(recipes),4),np.float64)
        for i,recipe in enumerate(recipes):
            predicted=select_refiner_instances(sets,recipe,model,True)
            for view in source["views"]:result[i]+=b.metric_counts(view["rles"],predicted)
        for candidates in sets.values():
            for c in candidates:c.pop("_decoded",None)
        return result
    counts=sum(b.bounded_map(score_source,manifest["splits"]["calibration"],4))
    grid=[dict(**recipe,**b.summarize(count)) for recipe,count in zip(recipes,counts)]
    grid.sort(key=lambda row:row["PQ"],reverse=True)
    baseline_counts=np.zeros(4,np.float64)
    for source in manifest["splits"]["calibration"]:
        predicted=[c["rle"] for c in data[source["stem"]]["baseline"]]
        for view in source["views"]:baseline_counts+=b.metric_counts(view["rles"],predicted)
    baseline=b.summarize(baseline_counts)["PQ"]
    chosen={k:grid[0][k] for k in ("family","threshold","cutoff","min_area")}
    if cfg["mode"]=="full" and grid[0]["PQ"]<baseline+cfg["minimum_calibration_gain"]:
        chosen=dict(family="prior")
    b.save_json(Path(cfg["run"])/"calibration_grid.json",grid)
    b.save_json(Path(cfg["run"])/"selected_refinement.json",dict(recipe=chosen,
        prior_calibration_PQ=baseline,refiner_best=grid[0],minimum_gain=cfg["minimum_calibration_gain"],
        prior_original_reported_calibration_PQ=manifest["prior_calibration"]["calibration"]["PQ"],
        report_used=False,note="Selection/checkpoint/quality calibration share dates; report is excluded from fitting"))
    print("Selected refinement:",json.dumps(chosen),"| new calibration PQ",grid[0]["PQ"],"| prior",baseline,flush=True)
    return chosen


def evaluate_refiner(cfg,folder,recipe):
    model=load_quality(Path(cfg["run"])/"refiner_quality.joblib")
    manifest=b.load_json(Path(cfg["run"])/"manifest.json")
    totals=np.zeros(4,np.float64);baseline=np.zeros(4,np.float64);dates=defaultdict(lambda:np.zeros(4,np.float64));rows=[]
    for source in manifest["splits"]["report"]:
        sets=b.load_json(Path(folder)/(source["stem"]+".json"))
        predictions=select_refiner_instances(sets,recipe,model)
        prior=[c["rle"] for c in sets["baseline"]]
        count=np.zeros(4,np.float64)
        for view in source["views"]:
            count+=b.metric_counts(view["rles"],predictions)
            baseline+=b.metric_counts(view["rles"],prior)
        totals+=count;dates[source["date"]]+=count
        rows.append(dict(stem=source["stem"],date=source["date"],predictions=len(predictions),**b.summarize(count)))
    samples=np.asarray(list(dates.values()));rng=np.random.default_rng(cfg["seed"])
    boot=[b.summarize(samples[rng.integers(len(samples),size=len(samples))].sum(0))["PQ"] for _ in range(1000)]
    result=dict(selected=b.summarize(totals),baseline=b.summarize(baseline),recipe=recipe,
        difference_PQ=b.summarize(totals)["PQ"]-b.summarize(baseline)["PQ"],
        date_bootstrap_95ci=list(map(float,np.quantile(boot,[.025,.975]))),images=len(rows),
        fingerprint=cfg["fingerprint"],public_PQ="not measured",report_used_for_fitting=False,
        note="Held-out within this run; same report dates have been inspected across prior experiments")
    b.save_json(Path(cfg["run"])/"refiner_report.json",result)
    b.save_json(Path(cfg["run"])/"report_by_image.json",rows)
    print("Refiner held-out report:",json.dumps(result,indent=2),flush=True)
    return result


def export_refiner(cfg,folder,recipe):
    run=Path(cfg["run"]);manifest=b.load_json(run/"manifest.json")
    model=load_quality(run/"refiner_quality.joblib")
    if cfg["mode"]=="full" and len(manifest["splits"]["test"])!=180:raise ValueError("Need all 180 test images")
    prefix="smoke_" if cfg["mode"]=="smoke" else ""
    files={"selected":run/(prefix+"submission.csv"),"baseline":run/(prefix+"submission_baseline.csv")}
    audits={}
    for kind,path in files.items():
        identifiers=set();per_image=[]
        with path.with_suffix(".tmp").open("w",newline="",encoding="utf8") as stream:
            writer=csv.writer(stream);writer.writerow(["filament_id","segmentation_rle"])
            for source in manifest["splits"]["test"]:
                sam_remaining(cfg,30)
                sets=b.load_json(Path(folder)/(source["stem"]+".json"))
                predictions=select_refiner_instances(sets,recipe if kind=="selected" else dict(family="prior"),model)
                owned=np.zeros((2048,2048),bool)
                for j,rle in enumerate(predictions,1):
                    mask=b.decode(rle)
                    if mask.shape!=(2048,2048) or not mask.any() or (owned&mask).any():raise ValueError("Invalid/overlapping output mask")
                    if b.encode(mask)["counts"]!=rle["counts"]:raise ValueError("RLE round trip failed")
                    owned|=mask;identity=f"{source['stem']}_{j}"
                    if identity in identifiers:raise ValueError("Duplicate output ID")
                    identifiers.add(identity);writer.writerow([identity,rle["counts"]])
                per_image.append(dict(image=source["name"],predictions=len(predictions)))
        os.replace(path.with_suffix(".tmp"),path)
        audits[kind]=dict(file=str(path),sha256=b.file_sha(path),masks=len(identifiers),images=len(per_image),
            per_image=per_image,roundtrip=True,nonoverlap=True,native_shape=[2048,2048])
    b.save_json(run/"submission_audits.json",audits)
    if cfg["mode"]=="full":shutil.copy2(files["selected"],Path(cfg["output_root"])/"submission.csv")
    b.save_json(run/"session_timing.json",dict(seconds=time.time()-cfg["session_started"],
        budget_seconds=cfg["total_hours"]*3600,prior_full_run_seconds=14816.277888,
        decoder_only_training=True,official_public_score="not measured"))
    print("Exported:",files,"| mask counts:",{k:v["masks"] for k,v in audits.items()},flush=True)
    return files


def refiner_main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    parser.add_argument("--task",choices=["cache","train","infer"],required=True)
    parser.add_argument("--slot",type=int,default=0);parser.add_argument("--part",default="test")
    args=parser.parse_args()
    try:
        if args.task=="cache":cache_sam_embeddings(args.config,args.slot)
        elif args.task=="train":train_sam_decoder(args.config)
        else:infer_sam_worker(args.config,args.part,args.slot)
    finally:
        if dist.is_initialized():dist.destroy_process_group()



if __name__ == '__main__':
    refiner_main()
