"""Build the portable full-frame notebook from its readable runtime module."""
import ast
import json
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip() + "\n"))


def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip() + "\n"))


md("""
# Solar filaments: native 2048 × 2048 U-Net

Train on **complete images** and predict each image in a single forward pass,
keeping narrow filaments at their original pixel resolution. This is a separate experiment from
`solar-filament-u-net.ipynb`; its tiled baseline is preserved.

The full-image U-Net uses seven downsampling stages (2048 → 16), a full-resolution
skip connection, and a small global-context branch. Width starts at 16 channels
and is capped at 256. Extra depth expands context; narrow early layers control
the activation memory cost. GroupNorm supports small batches without running
statistics, and activation checkpointing trades extra computation for memory.

Target: **2 GPUs with 14 GiB each, 30 GB host RAM, four total loader workers**.
DDP trains a replica on each GPU using different **whole images**; GPU memories
are separate. FP16 AMP, a real forward/backward/Adam memory probe, and gradient
accumulation make this practical without assuming a measured VRAM requirement.

Thin-structure safeguards: no resizing or crop training, pixel-preserving flips
and 90° rotations, full-resolution skips, and thresholding/component extraction
only after a complete-image prediction. The component-area filter defaults to
one pixel. Validation must decide whether larger filtering improves quality.
These choices remove artificial tile seams; they do not guarantee connectivity
or resolve two distinct filaments that touch in a binary union mask.
""")

code("""
%pip install -q pycocotools scikit-learn pandas matplotlib
""")

code("""
import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
import csv
import gc
import hashlib
import importlib.util
import json
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from sklearn.model_selection import train_test_split

cv2.setNumThreads(1)
torch.set_num_threads(1)
DATA_ROOT_OVERRIDE = None
ANNOTATION_FILENAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_SIZE = 2048
RANDOM_SEED = 10

def is_dataset_root(path):
    return ((path / "train" / "train_images").is_dir()
            and (path / "train" / ANNOTATION_FILENAME).is_file()
            and (path / "test" / "test_images").is_dir())

def find_dataset_root():
    candidates = []
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidates.extend((parent, parent / "MAGFilo_1.0_Kaggle_2026",
                           parent / "MAGFiLO_1.0_Kaggle_2026"))
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates.extend(p.parent.parent for p in kaggle.rglob(ANNOTATION_FILENAME))
    for candidate in candidates:
        if is_dataset_root(candidate):
            return candidate.resolve()
    raise FileNotFoundError("Set DATA_ROOT_OVERRIDE to the folder containing train/ and test/.")

DATA_ROOT = (Path(DATA_ROOT_OVERRIDE).expanduser().resolve()
             if DATA_ROOT_OVERRIDE else find_dataset_root())
if not is_dataset_root(DATA_ROOT):
    raise FileNotFoundError(f"Invalid dataset root: {DATA_ROOT}")
TRAIN_IMAGES_DIR = DATA_ROOT / "train" / "train_images"
TEST_IMAGES_DIR = DATA_ROOT / "test" / "test_images"
ANNOTATION_PATH = DATA_ROOT / "train" / ANNOTATION_FILENAME
OUTPUT_DIR = Path.cwd() / "unet_full_image_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUTPUT_DIR / "full_image_unet_best.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"
SUBMISSION_PATH = OUTPUT_DIR / "solar_filament_full_image_submission.csv"
print(f"Dataset: {DATA_ROOT}")
""")

md("""
## Keep source files separated between training and validation

Use the baseline's split: multi-annotator source files stay in training, and
validation contains only single-annotator files. Each annotator record remains
an individual training example. This preserves the existing label policy for
comparison, including its extra weight on multiply annotated sources.
""")

code("""
annotation_bytes = ANNOTATION_PATH.read_bytes()
annotations = json.loads(annotation_bytes)
image_by_id = {item["id"]: item for item in annotations["images"]}
annotations_by_image = defaultdict(list)
for annotation in annotations["annotations"]:
    annotations_by_image[annotation["image_id"]].append(annotation)
name_to_image_ids = defaultdict(list)
for item in annotations["images"]:
    if (int(item["height"]), int(item["width"])) != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected native 2048-square annotations: {item['file_name']}")
    name_to_image_ids[item["file_name"]].append(item["id"])

single_ids = [ids[0] for ids in name_to_image_ids.values() if len(ids) == 1]
multi_ids = [image_id for ids in name_to_image_ids.values() if len(ids) > 1 for image_id in ids]
if len(single_ids) < 2:
    raise ValueError("At least two single-annotator images are required.")
val_count = min(max(1, round(0.2 * len(name_to_image_ids))), len(single_ids) - 1)
train_single_ids, val_image_ids = train_test_split(
    single_ids, test_size=val_count, random_state=RANDOM_SEED, shuffle=True)
train_image_ids = train_single_ids + multi_ids
train_files = {image_by_id[i]["file_name"] for i in train_image_ids}
val_files = {image_by_id[i]["file_name"] for i in val_image_ids}
assert not train_files.intersection(val_files), "Source-image leakage"
print(f"Train: {len(train_image_ids)} records / {len(train_files)} source files")
print(f"Validation: {len(val_image_ids)} complete images")

def polygons_to_mask(image_id):
    info = image_by_id[image_id]
    mask = np.zeros((int(info["height"]), int(info["width"])), dtype=np.uint8)
    for annotation in annotations_by_image[image_id]:
        segmentation = annotation["segmentation"]
        if isinstance(segmentation, dict):
            rle = segmentation
            if isinstance(rle["counts"], list):
                rle = mask_utils.frPyObjects(rle, *mask.shape)
            decoded = mask_utils.decode(rle)
            if decoded.ndim == 3:
                decoded = decoded.any(axis=2)
            mask |= (decoded > 0).astype(np.uint8)
        else:
            for polygon in segmentation:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask
""")

md("""
## Stream complete masks to disk

Read original grayscale images on demand and cache only compressed uint8 masks.
No tile cache or per-process copy of the entire dataset is created. With four
training workers, prefetch factor one, and batch one or two, queued image/mask
arrays remain small relative to 30 GB RAM. CPU decoding and storage speed may
still limit GPU utilization; the runtime reports image throughput and peak VRAM.
""")

code("""
mask_fingerprint = hashlib.sha256(annotation_bytes + b"full-image-cv2-union-v1").hexdigest()[:16]
MASK_DIR = OUTPUT_DIR / "masks" / mask_fingerprint
MASK_DIR.mkdir(parents=True, exist_ok=True)

def make_records(image_ids):
    records = []
    for index, image_id in enumerate(image_ids, start=1):
        info = image_by_id[image_id]
        image_path = (TRAIN_IMAGES_DIR / info["file_name"]).resolve()
        with Image.open(image_path) as opened:
            if opened.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Image is not native 2048-square: {image_path}")
        safe_name = hashlib.sha256(str(image_id).encode()).hexdigest()
        mask_path = (MASK_DIR / f"{safe_name}.png").resolve()
        if not mask_path.exists():
            if not cv2.imwrite(str(mask_path), polygons_to_mask(image_id) * 255):
                raise OSError(f"Cannot write mask: {mask_path}")
        records.append(dict(image_path=str(image_path), mask_path=str(mask_path),
                            image_id=image_id, file_name=info["file_name"]))
        if index % 100 == 0 or index == len(image_ids):
            print(f"Prepared {index}/{len(image_ids)} full-image records", flush=True)
    return records

train_records = make_records(train_image_ids)
val_records = make_records(val_image_ids)
(OUTPUT_DIR / "split_manifest.json").write_text(
    json.dumps(dict(train=train_records, val=val_records, image_size=IMAGE_SIZE), indent=2),
    encoding="utf-8")
""")

code("""
# Inspect native detail as well as the complete image. This crop is visualization only.
sample = val_records[0]
sample_image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
sample_mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
ys, xs = np.where(sample_mask)
cy, cx = ((int(np.median(ys)), int(np.median(xs))) if len(ys) else (1024, 1024))
y0, x0 = min(max(cy - 128, 0), IMAGE_SIZE - 256), min(max(cx - 128, 0), IMAGE_SIZE - 256)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(sample_image, cmap="gray")
axes[0].imshow(np.ma.masked_where(~sample_mask, sample_mask), cmap="autumn", alpha=0.6)
axes[0].set_title("Complete annotated image")
axes[1].imshow(sample_image[y0:y0+256, x0:x0+256], cmap="gray", interpolation="nearest")
axes[1].set_title("Native pixels: 256 × 256 detail")
axes[2].imshow(sample_mask[y0:y0+256, x0:x0+256], cmap="gray", interpolation="nearest")
axes[2].set_title("Thin structures in target")
for axis in axes:
    axis.axis("off")
plt.tight_layout()
plt.show()
del sample_image, sample_mask, ys, xs
""")

md("""
## Training settings and memory probe

Start with base width 16, depth 7, FP16, and checkpointing. The GPU probe tries
microbatches 1 through 4 using real backward and optimizer steps, restores initial
weights/RNG, and chooses the fastest common fit with reserved-memory headroom.
It does not combine the GPUs' VRAM. Leave additional headroom if other processes
use the GPUs. A startup probe reduces OOM risk; the target run must still verify
actual training peaks. To recover from OOM, first set `PER_GPU_BATCH_SIZE = 1`;
if necessary set `BASE_CHANNELS = 8`. Keep the native image resolution.

Effective batch = microbatch × GPU count × accumulation steps (8 at batch one,
two GPUs, accumulation four). Dice is computed for each distributed microbatch;
accumulating gradients does not turn it into a single Dice over all eight images.
Epoch checkpoint selection retains the baseline's combined loss with hard Dice;
it is a pixel/union objective, not a connectivity or exact instance-PQ objective.

GroupNorm and global-context features couple distant regions, while decoder
skips retain native detail. Additional layers alone do not restore missing fine
features. Inspect the native-detail validation plots and complete-image metrics
before claiming an improvement over the tiled model.
""")

code("""
BASE_CHANNELS = 16
DEPTH = 7
MAX_CHANNELS = 256
PER_GPU_BATCH_SIZE = "auto"     # Probe 1 through 4; set integer 1 for a conservative run.
GRADIENT_ACCUMULATION_STEPS = 4
NUM_WORKERS = 4                # TOTAL: two per rank with two GPUs.
PREFETCH_FACTOR = 1
MEMORY_FRACTION = 0.85
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LOSS_WEIGHTS = {"log_soft_pq": 0.4, "dice": 0.4, "bce": 0.2}
AMP_ENABLED = True
USE_CHANNELS_LAST = True
NUM_GPUS = min(2, torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("Select a CUDA-enabled PyTorch kernel or Kaggle GPU accelerator.")
if NUM_GPUS > 1 and not sys.platform.startswith("linux"):
    raise RuntimeError("The two-GPU NCCL training path requires Linux/Kaggle. A single GPU can smoke-test locally.")
DEVICE = torch.device("cuda:0")
for index in range(NUM_GPUS):
    properties = torch.cuda.get_device_properties(index)
    print(f"GPU {index}: {properties.name}, {properties.total_memory / 2**30:.2f} GiB")
print(f"{NUM_GPUS} GPU(s); {NUM_WORKERS} total workers; no resize or tiling")
""")

runtime = (ROOT / "solar_unet_full_image_runtime.py").read_text(encoding="utf-8")
code("# Embedded worker: this notebook can be uploaded to Kaggle on its own.\n"
     "FULL_IMAGE_RUNTIME_SOURCE = " + repr(runtime) + "\n"
     "runtime_path = OUTPUT_DIR / 'solar_unet_full_image_runtime.py'\n"
     "runtime_path.write_text(FULL_IMAGE_RUNTIME_SOURCE, encoding='utf-8')\n"
     "runtime_spec = importlib.util.spec_from_file_location('solar_unet_full_image_runtime', runtime_path)\n"
     "full_runtime = importlib.util.module_from_spec(runtime_spec)\n"
     "sys.modules[runtime_spec.name] = full_runtime\n"
     "runtime_spec.loader.exec_module(full_runtime)\n")

code("""
runtime_config = dict(
    train_records=train_records, val_records=val_records, image_size=IMAGE_SIZE,
    base_channels=BASE_CHANNELS, depth=DEPTH, max_channels=MAX_CHANNELS,
    checkpoint_blocks=True, per_gpu_batch_size=PER_GPU_BATCH_SIZE,
    batch_candidates=[1, 2, 3, 4], memory_fraction=MEMORY_FRACTION,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    seed=RANDOM_SEED, epochs=NUM_EPOCHS, patience=EARLY_STOPPING_PATIENCE,
    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, loss_weights=LOSS_WEIGHTS,
    amp=AMP_ENABLED, workers=NUM_WORKERS, prefetch_factor=PREFETCH_FACTOR,
    channels_last=USE_CHANNELS_LAST, cudnn_benchmark=False, log_every=10,
    checkpoint_path=str(CHECKPOINT_PATH), history_path=str(HISTORY_PATH),
    runtime_report_path=str(OUTPUT_DIR / "runtime_report.json"))
config_path = OUTPUT_DIR / "training_config.json"
config_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")

# Release a previous notebook model before starting the independent worker processes.
for variable in ("model", "checkpoint", "optimizer", "scaler"):
    globals().pop(variable, None)
gc.collect()
torch.cuda.empty_cache()
command = [sys.executable, "-u"]
if NUM_GPUS > 1:
    command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={NUM_GPUS}"]
command += [str(runtime_path), "--config", str(config_path)]
worker_environment = os.environ.copy()
worker_environment.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                          MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
visible_mask = worker_environment.get("CUDA_VISIBLE_DEVICES")
visible_devices = visible_mask.split(",") if visible_mask else [str(i) for i in range(NUM_GPUS)]
worker_environment["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices[:NUM_GPUS])
log_path = OUTPUT_DIR / "distributed_training.log"
training_started = time.perf_counter()
with log_path.open("w", encoding="utf-8") as log_file:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, env=worker_environment,
                               start_new_session=(os.name != "nt"),
                               creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    finally:
        if process.poll() is None:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait()
if return_code:
    raise RuntimeError(f"Training exited with code {return_code}. See {log_path}.")
print(f"Training elapsed: {(time.perf_counter() - training_started) / 60:.1f} min")
""")

code("""
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
model = full_runtime.FullImageUNet(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
USE_CHANNELS_LAST = bool(checkpoint.get("channels_last", True))
if USE_CHANNELS_LAST:
    model.to(memory_format=torch.channels_last)
model.eval()
history_df = pd.read_csv(HISTORY_PATH)
display(history_df.tail())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train")
axes[0].plot(history_df["epoch"], history_df["val_loss"], label="validation")
axes[0].set_ylabel("Composite loss")
axes[0].legend()
axes[1].plot(history_df["epoch"], history_df["val_dice_at_0_5"])
axes[1].set_ylabel("Full-image validation Dice at 0.5")
for axis in axes:
    axis.set_xlabel("Epoch")
plt.tight_layout()
plt.show()
""")

md("""
## Select a threshold using complete images

One native-resolution forward per validation image; connected components cross
the former 512-pixel grid freely. Optimize the harmonic mean of pixel Dice and
connected-component PQ, keeping the baseline's threshold-selection rule but
evaluating complete images. Component PQ on union labels is a **proxy**, since
touching polygon instances merge. The training log-soft-PQ term is a smooth
union-overlap surrogate and does not match instances or enforce topology.

`MIN_COMPONENT_AREA = 1` retains narrow and small predictions. Tune it using
validation before filtering tiny components. No morphological closing is applied:
it can join nearby distinct filaments. This evaluation uses GPU 0, batch one;
both GPUs participate in the distributed training above.
""")

code("""
THRESHOLD_CANDIDATES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
MIN_COMPONENT_AREA = 1
PQ_MATCH_IOU = 0.50

def component_map(binary, min_area=1):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(binary, dtype=np.uint8), connectivity=8)
    keep = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    remap = np.zeros(count, dtype=np.int32)
    remap[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    return remap[labels], stats[keep, cv2.CC_STAT_AREA].astype(np.int64)

def panoptic_quality_counts(prediction, target_components, predicted_components=None):
    target_labels, target_areas = target_components
    predicted_labels, predicted_areas = (component_map(prediction, MIN_COMPONENT_AREA)
                                         if predicted_components is None else predicted_components)
    nt, npred = len(target_areas), len(predicted_areas)
    valid = (target_labels > 0) & (predicted_labels > 0)
    # Sparse label-pair intersections avoid a potentially enormous nt × npred matrix.
    codes = target_labels[valid].astype(np.int64) * (npred + 1) + predicted_labels[valid]
    pair, intersection = np.unique(codes, return_counts=True)
    ti, pi = pair // (npred + 1) - 1, pair % (npred + 1) - 1
    union = target_areas[ti] + predicted_areas[pi] - intersection
    iou = intersection / np.maximum(union, 1)
    match = iou > PQ_MATCH_IOU
    # With disjoint components and strict IoU > 0.5, matches are unique.
    tp = int(match.sum())
    return dict(matched_iou=float(iou[match].sum()), tp=tp, fp=npred-tp, fn=nt-tp)

threshold_totals = {t: dict(intersection=0, predicted_pixels=0, target_pixels=0,
                           matched_iou=0.0, tp=0, fp=0, fn=0) for t in THRESHOLD_CANDIDATES}
threshold_started = time.perf_counter()
for index, record in enumerate(val_records, start=1):
    image = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
    probabilities = full_runtime.predict_full_image(
        model, image, DEVICE, amp=AMP_ENABLED, channels_last=USE_CHANNELS_LAST)
    target_components = component_map(target)  # retain every annotated pixel
    for threshold, totals in threshold_totals.items():
        raw_prediction = probabilities >= threshold
        # Apply exactly the same filtering to pixel metrics, PQ and submission.
        predicted_components = component_map(raw_prediction, MIN_COMPONENT_AREA)
        predicted_labels, _ = predicted_components
        prediction = predicted_labels > 0
        totals["intersection"] += int(np.logical_and(prediction, target).sum())
        totals["predicted_pixels"] += int(prediction.sum())
        totals["target_pixels"] += int(target.sum())
        for key, value in panoptic_quality_counts(prediction, target_components, predicted_components).items():
            totals[key] += value
    if index % 20 == 0 or index == len(val_records):
        print(f"Full-image threshold validation: {index}/{len(val_records)}", flush=True)

threshold_rows = []
for threshold, totals in threshold_totals.items():
    denominator = totals["tp"] + 0.5 * (totals["fp"] + totals["fn"])
    pq = totals["matched_iou"] / denominator if denominator else 1.0
    dice = (2 * totals["intersection"] + 1.0) / (totals["predicted_pixels"] + totals["target_pixels"] + 1.0)
    joint = 2 * pq * dice / (pq + dice) if pq + dice else 0.0
    threshold_rows.append(dict(threshold=threshold, validation_joint_score=joint,
                               validation_component_PQ=pq, validation_pixel_Dice=dice,
                               TP=totals["tp"], FP=totals["fp"], FN=totals["fn"]))
threshold_summary = pd.DataFrame(threshold_rows).sort_values(
    ["validation_joint_score", "validation_component_PQ", "validation_pixel_Dice", "threshold"],
    ascending=[False, False, False, True]).reset_index(drop=True)
BEST_THRESHOLD = float(threshold_summary.iloc[0]["threshold"])
threshold_summary.to_csv(OUTPUT_DIR / "threshold_validation.csv", index=False)
(OUTPUT_DIR / "inference_settings.json").write_text(json.dumps(
    dict(threshold=BEST_THRESHOLD, min_component_area=MIN_COMPONENT_AREA, image_size=IMAGE_SIZE), indent=2),
    encoding="utf-8")
display(threshold_summary)
print(f"Threshold {BEST_THRESHOLD:.2f}; {(time.perf_counter()-threshold_started)/60:.1f} min")
""")

code("""
def predict_full_resolution(image, threshold=BEST_THRESHOLD):
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between zero and one.")
    probabilities = full_runtime.predict_full_image(
        model, image, DEVICE, amp=AMP_ENABLED, channels_last=USE_CHANNELS_LAST)
    labels, _ = component_map(probabilities >= threshold, MIN_COMPONENT_AREA)
    return labels > 0

record = val_records[0]
image = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
target = cv2.imread(record["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
prediction = predict_full_resolution(image)
dice = (2 * np.logical_and(prediction, target).sum() + 1) / (prediction.sum() + target.sum() + 1)
ys, xs = np.where(target)
cy, cx = ((int(np.median(ys)), int(np.median(xs))) if len(ys) else (1024, 1024))
y0, x0 = min(max(cy - 128, 0), IMAGE_SIZE - 256), min(max(cx - 128, 0), IMAGE_SIZE - 256)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for row, (y, x, size) in enumerate(((0, 0, IMAGE_SIZE), (y0, x0, 256))):
    for col, (data, title) in enumerate(((image, "Image"), (target, "Target"), (prediction, "Prediction"))):
        axes[row, col].imshow(data[y:y+size, x:x+size], cmap="gray", interpolation="nearest")
        axes[row, col].set_title(title + (" — native detail" if row else ""))
        axes[row, col].axis("off")
fig.suptitle(f"Complete-image Dice: {dice:.4f}; inspect thin branches and gaps")
plt.tight_layout()
plt.show()
""")

md("""
## Submission from full-image components

Run a single full-image forward, then find components and encode each in COCO
compressed RLE, using the baseline's `filament_id,segmentation_rle` format.
Process one component at a time to bound RAM even when many small components
survive. An empty prediction emits no filament rows, matching the original
notebook. The schema check below validates encoding; competition scoring still
needs the actual submission evaluation.
""")

code("""
CREATE_SUBMISSION = True

def mask_to_rle(binary):
    encoded = mask_utils.encode(np.asfortranarray(binary, dtype=np.uint8))
    return encoded["counts"].decode("utf-8")

def iter_prediction_rows(stem, binary):
    labels, areas = component_map(binary, MIN_COMPONENT_AREA)
    # Stable ordering: descending area, then component scan order.
    for index, label in enumerate(np.argsort(-areas, kind="stable") + 1, start=1):
        yield f"{stem}_{index}", mask_to_rle(labels == label)

def create_submission(output_path=SUBMISSION_PATH):
    paths = sorted((p for p in TEST_IMAGES_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No test images in {TEST_IMAGES_DIR}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise ValueError("Test files have duplicate filename stems.")
    row_count = 0
    started = time.perf_counter()
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filament_id", "segmentation_rle"])
        for index, path in enumerate(paths, start=1):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Unreadable or incorrectly sized test image: {path}")
            prediction = predict_full_resolution(image)
            for row in iter_prediction_rows(path.stem, prediction):
                writer.writerow(row)
                row_count += 1
            if index % 20 == 0 or index == len(paths):
                print(f"Inference: {index}/{len(paths)} images; {row_count} filament rows", flush=True)
    print(f"Saved {output_path}; {(time.perf_counter()-started)/60:.1f} min")
    return Path(output_path)

if CREATE_SUBMISSION:
    create_submission()
""")

code("""
if CREATE_SUBMISSION:
    seen = set()
    checked = 0
    with SUBMISSION_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["filament_id", "segmentation_rle"]
        for row in reader:
            assert row["filament_id"] and row["segmentation_rle"]
            assert row["filament_id"] not in seen, "Duplicate filament ID"
            seen.add(row["filament_id"])
            if checked < 3:
                decoded = mask_utils.decode(dict(size=[IMAGE_SIZE, IMAGE_SIZE],
                                                 counts=row["segmentation_rle"].encode("utf-8")))
                assert decoded.shape == (IMAGE_SIZE, IMAGE_SIZE) and decoded.any()
                checked += 1
    print(f"Submission schema verified: {len(seen)} rows, {checked} sample RLEs decoded")
""")

md("""
## Outputs and references

`unet_full_image_outputs/` contains the best checkpoint with architecture
settings, training history, GPU probe/peak-memory report, logs, split manifest,
compressed full-image masks, selected threshold, and competition submission.
The runtime is embedded in this notebook so no Python file must be uploaded
separately. The readable source is `solar_unet_full_image_runtime.py`.

Actual throughput, VRAM fit, and segmentation improvement must be measured on
the two target GPUs. Do not interpret higher memory occupancy as better training.
If wider early features are affordable after measurement, compare base width
24 or 32 on validation; native resolution is the first priority for thin filaments.

PyTorch primary references:
[DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
[activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html),
[AMP and accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html).
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "file_extension": ".py"},
})
nbf.validate(notebook)
for cell in cells:
    if cell.cell_type == "code" and not cell.source.startswith("%"):
        ast.parse(cell.source)
target = ROOT / "solar-filament-u-net-full-image.ipynb"
nbf.write(notebook, target)
print(f"Wrote {target.name}: {len(cells)} cells; embedded runtime {len(runtime):,} characters")
