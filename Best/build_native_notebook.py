"""Rebuild the standalone notebook after editing native_yolo_runtime.py."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
runtime = (HERE / "native_yolo_runtime.py").read_text(encoding="utf-8")
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md('''
# Solar filaments — native 2048 × 2048 YOLO26 segmentation

This is the native-resolution successor to `solar-filament-yolo26-full-res-0-36.ipynb`.
The original notebook and its `results/` remain the saved baseline. This keeps the
**YOLO26s segmentation model**, AdamW, cosine decay, seed 42, confidence search and
connected-component submission policy. SAM is not added to this pipeline.

**No input resizing, crops, tiles, letterboxing, spatial scaling or interpolated
rotation.** Every source and every training target must be 2048 × 2048; other shapes
raise an error. Only exact pixel flips and brightness augmentation are enabled.
All annotator records for one filename contribute to one native pixelwise union.
8-connected components become exact training instance-ID masks, including one-pixel
components. No contour approximation or YOLO polygon round trip is used.

The mask loss upsamples predicted logits to the unchanged target grid. YOLO still
has strided internal features and lower-resolution mask prototypes; native input
does not make every internal feature map 2048 × 2048. Standard box-based assignment
replaces YOLO26's mask-based assignment so it cannot downsample the target masks.
This is a deliberate training change, to be compared using validation PQ.

**Start with `MODE="smoke"` on Kaggle GPU T4 ×2**, inspect the merge/preprocessing
previews, tensor audits and logs, then set `MODE="full"`. Four preprocessing variants
train separately, sequentially; each training job uses both GPUs with DDP. Validation
and test prediction use one replica per GPU on disjoint source shards. The notebook
is standalone: its embedded runtime is written automatically, including in Kaggle.
''')
code('''
from pathlib import Path
import os

MODE = "smoke"                 # "smoke": 8 sources per partition, 1 epoch; "full": entire split
CFG = {
    "data_root": "",           # folder containing train/ and test/; empty = local/Kaggle discovery
    "output_root": "/kaggle/working" if Path("/kaggle").exists() else "./native_outputs",
    "run_name": "yolo26s_native_union_b2",  # fresh directory for the larger per-GPU batch
    "gpu_ids": [0, 1],          # two T4s; explicitly set [0] for a single-GPU smoke check
    "cpu_workers": 4,          # TOTAL CPU pool workers or training loader workers across GPUs
    "vram_budget_gib": 13.0,   # per GPU; memory is separate on each GPU
    "seed": 42,
    "val_fraction": 0.20,
    "weights": "yolo26s-seg.pt",  # alternatively the absolute saved-baseline best.pt path
    "epochs": 150,
    "patience": 20,
    "batch_per_gpu": 2,        # global batch 4 on 2 GPUs; no automatic image-size reduction
    "nominal_batch": 64,       # Ultralytics gradient accumulation target
    "lr0": 0.001,
    "amp": True,
    "fliplr": 0.5,
    "flipud": 0.5,
    "brightness": 0.3,
    "merge": {"min_area": 1, "connectivity": 8},
    "prediction_min_area": 1,  # use 20 to compare the previous run's component-area filter
    "preprocessing": {
        "clahe_clip": 2.0, "clahe_grid": [8, 8],
        "percentiles": [1.0, 99.0], "gamma": 0.8,
    },
    "confidences": [0.15, 0.25, 0.35, 0.45, 0.55],
    "max_det": 100,
    "smoke_sources": 8,         # 4 sources/rank: exercises two real batches of 2 on each GPU
    "plots": True,
    "resume": False,          # recover an interrupted run using its last.pt
}
VARIANTS = ["raw", "clahe", "percentile", "gamma"]  # choose one or several; independent models
CREATE_SUBMISSION = True       # only the variant with the highest validation PQ processes test
''')
md('''
## Dependencies and resources

Keep Kaggle's CUDA PyTorch/torchvision installation. This installs only missing
auxiliary packages and requires `ultralytics==8.4.198` because the native trainer
uses its loss and validation interfaces. If an incompatible SDK is already loaded,
install the required version and restart the kernel. Internet is needed once for
packages/pretrained weights; local weight paths support offline runs.

Preparation and CPU scoring use four bounded threads. DDP has two loader workers
per rank and no extra validation loader workers. Inference has two CPU threads per
GPU for prefetch/encoding. Numerical library threads are capped at one. Checkpointed
per-anchor mask loss trades computation for bounded native-mask activation memory.
GPU VRAM is **not pooled**; each GPU must fit its own full image and model.

Training defaults to two images per GPU, four total. The eight-source smoke split
has two full batches per rank, so it exercises the actual batch size and loader
workers. If you reduce smoke sources to two, each rank only receives one image and
the SDK may print `Using 0 dataloader workers`; that does not measure batch-two memory.
Training requests two loader workers per rank. Pretrained weights are resolved once
in the notebook coordinator before DDP starts; both ranks receive the same absolute checkpoint path.
Optional Ray Tune tracking is disabled, and prediction uses `quantize=16` for FP16.

`native_memory_rank*.json` records peak allocated/reserved training memory and actual
batch sizes for each GPU. The allocator budget remains 13 GiB per T4, leaving room
for CUDA and distributed communication. The earlier one-image smoke log reported
5.26 GiB, suggesting room for two; fitting two or more on dense images still requires
the Kaggle smoke check. Validation and prediction process one image per GPU at a time.
''')
code('''
import importlib.util
import importlib.metadata
import subprocess
import sys
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
packages = {"ultralytics": "ultralytics==8.4.198", "pycocotools": "pycocotools>=2.0.7,<3",
            "scipy": "scipy>=1.11,<2", "cv2": "opencv-python-headless>=4.8,<6",
            "pandas": "pandas>=2.1,<4", "matplotlib": "matplotlib>=3.8,<4"}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing])
if importlib.metadata.version("ultralytics") != "8.4.198":
    raise RuntimeError("Install ultralytics==8.4.198, restart the kernel, then run all cells.")

import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display

assert MODE in ("smoke", "full")
assert VARIANTS and len(set(VARIANTS)) == len(VARIANTS)
assert set(VARIANTS) <= {"raw", "clahe", "percentile", "gamma"}
assert CFG["gpu_ids"] and len(set(CFG["gpu_ids"])) == len(CFG["gpu_ids"])
assert torch.cuda.is_available() and min(CFG["gpu_ids"]) >= 0 and max(CFG["gpu_ids"]) < torch.cuda.device_count(), "Select GPU T4 x2, or explicitly configure gpu_ids=[0]."
assert CFG["cpu_workers"] >= len(CFG["gpu_ids"]) and CFG["batch_per_gpu"] >= 1
assert CFG["merge"]["connectivity"] in (4, 8) and CFG["merge"]["min_area"] >= 1
assert 0 < CFG["val_fraction"] < 1 and CFG["prediction_min_area"] >= 1
assert CFG["preprocessing"]["gamma"] > 0
assert 0 <= CFG["preprocessing"]["percentiles"][0] < CFG["preprocessing"]["percentiles"][1] <= 100
assert all(0 < c < 1 for c in CFG["confidences"])
CFG["mode"] = MODE
RUN = Path(CFG["output_root"]).expanduser().resolve() / f"{CFG['run_name']}_{MODE}"
RUN.mkdir(parents=True, exist_ok=True)
for gpu in CFG["gpu_ids"]:
    info = torch.cuda.get_device_properties(gpu)
    print(f"GPU {gpu}: {info.name}, {info.total_memory / 2**30:.1f} GiB")
print(f"Native input: 2048x2048; global batch: {len(CFG['gpu_ids']) * CFG['batch_per_gpu']}")
print(f"Output: {RUN}; free disk: {shutil.disk_usage(RUN).free / 2**30:.1f} GiB")
''')
md('''
## Embedded native runtime

Run this cell unchanged. It writes an importable module so fresh DDP ranks, loader
workers and prediction processes can use the exact same implementation. The module
is also required when loading these custom-model checkpoints in another session;
rerunning this notebook cell recreates it.
''')
code("RUNTIME_SOURCE = r'''\n" + runtime + "\n'''\n" + '''
runtime_dir = RUN / "runtime"
runtime_dir.mkdir(parents=True, exist_ok=True)
RUNTIME_PATH = runtime_dir / "native_yolo_runtime.py"
RUNTIME_PATH.write_text(RUNTIME_SOURCE, encoding="utf-8")
sys.path.insert(0, str(runtime_dir))
sys.modules.pop("native_yolo_runtime", None)
import native_yolo_runtime as native
native.save_json(RUN / "environment.json", {"python": sys.version, "torch": torch.__version__,
    "ultralytics": importlib.metadata.version("ultralytics"), "cuda": torch.version.cuda,
    "gpu_ids": CFG["gpu_ids"], "runtime_sha": native.file_sha(RUNTIME_PATH)})
''')
md('''
## Merge annotators and split independent sources

Grouping uses exact source filenames, as in the custom YOLO+SAM notebook. The union
can join touching annotations or split disconnected ones; it represents foreground
coverage, not inferred annotator instance identity. No dilation is used.
Native masks bypass polygon conversion, so thin and one-pixel components survive.
Pixel-identical sources with different filenames also stay together in the split.
All variants use this one split and these same targets.

Validation retains **each original annotator's instance masks**. Metrics are first
averaged across annotators for a source, then equally across physical sources. The
changed split/target and averaging policy mean the new scores should be compared
across these experiments; they are not a direct replication of the old saved score.
''')
code('''
MANIFEST_PATH = native.prepare_data(CFG, RUN)
manifest = native.load_json(MANIFEST_PATH)
assert not ({r["pixel_hash"] for r in manifest["train"]} & {r["pixel_hash"] for r in manifest["val"]})
native.save_json(RUN / "config.json", {**CFG, "variants": VARIANTS})
display(native.load_json(RUN / "data_audit.json"))
''')
code('''
# Visual audit: independent annotators, the merged native target, and intensity-only variants.
record = next((r for r in manifest["train"] if len(r["image_ids"]) > 1), manifest["train"][0])
image = native.read_gray(record["path"])
payload = native.load_json(manifest["annotation_path"])
with np.load(record["target"]) as target:
    labels = target["labels"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for axis in axes:
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.axis("off")
axes[0].set_title(record["name"])
colors = plt.get_cmap("tab10")
for i, image_id in enumerate(record["image_ids"]):
    annotator = np.zeros_like(labels, dtype=bool)
    for annotation in payload["annotations"]:
        if str(annotation["image_id"]) == image_id:
            annotator |= native.annotation_mask(annotation)
    if annotator.any():
        axes[1].contour(annotator, levels=[0.5], colors=[colors(i % 10)], linewidths=0.6)
axes[1].set_title(f"{len(record['image_ids'])} annotator records")
axes[2].imshow(np.ma.masked_where(labels == 0, labels), cmap="turbo", alpha=0.7)
axes[2].set_title(f"Native union: {len(record['boxes'])} instances")
fig.tight_layout(); fig.savefig(RUN / "merge_preview.png", dpi=150, bbox_inches="tight"); plt.show()
fig, axes = plt.subplots(1, len(VARIANTS), figsize=(6 * len(VARIANTS), 6), squeeze=False)
for axis, variant in zip(axes[0], VARIANTS):
    processed = native.preprocess(image, variant, CFG["preprocessing"])
    axis.imshow(processed, cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"{variant}: {processed.shape}"); axis.axis("off")
fig.tight_layout(); fig.savefig(RUN / "preprocessing_preview.png", dpi=150, bbox_inches="tight"); plt.show()
del labels, payload, image
''')
md('''
## Train experiments with both GPUs, then compare native validation PQ

Each variant starts from the same pretrained weights and has its own checkpoint,
logs, native-tensor audit, history, threshold trials and prediction cache. No
preprocessed image copies or full RGB `.npy` caches are materialized. Completed,
unchanged training jobs are reused; interrupted jobs require `resume=True` or a
fresh run name. An incompatible contract is rejected to prevent stale experiments.
When replacing this notebook/runtime in a directory created by an earlier revision,
use a new run name. Switching from smoke to full mode already selects a separate directory.

The raw model is the control. CLAHE changes local intensity contrast; percentile
stretch uses nonblack pixels; gamma uses an intensity lookup table. The same chosen
transform is applied during training, validation and test. CLAHE's grid partitions
its contrast calculation; it does not crop or resize the model's input image.
''')
code('''
jobs, scores = {}, []
for variant in VARIANTS:
    experiment = RUN / variant
    experiment.mkdir(parents=True, exist_ok=True)
    job_path = experiment / "job.json"
    job = {"cfg": CFG, "variant": variant, "manifest": str(MANIFEST_PATH), "run": str(experiment)}
    contract = native.digest({"job": {**job, "cfg": {k: v for k, v in CFG.items() if k != "resume"}},
                              "dataset": manifest["key"], "runtime": native.file_sha(RUNTIME_PATH)})
    contract_path = experiment / "contract.json"
    if contract_path.exists() and native.load_json(contract_path)["key"] != contract:
        raise RuntimeError(f"Incompatible settings for {variant}; use a new run_name.")
    native.save_json(contract_path, {"key": contract})
    native.save_json(job_path, job)
    marker = experiment / "training_complete.json"
    checkpoint_path = experiment / "weights/best.pt"
    completed = marker.exists() and checkpoint_path.exists()
    if completed and native.load_json(marker)["checkpoint_sha"] != native.file_sha(checkpoint_path):
        raise RuntimeError(f"Checkpoint changed for {variant}; use a fresh run_name.")
    if not completed:
        if (experiment / "weights/last.pt").exists() and not CFG["resume"]:
            raise RuntimeError(f"Interrupted {variant}: set resume=True or use a fresh run_name.")
        native.launch_training(job_path, RUNTIME_PATH)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        native.save_json(marker, {"checkpoint_sha": native.file_sha(checkpoint_path), "contract": contract})
    else:
        print(f"Reusing completed {variant} training")
    native.launch_prediction(job_path, RUNTIME_PATH, "val")
    scores.append(native.evaluate_job(job_path))
    jobs[variant] = job_path
    display(pd.DataFrame(scores))
leaderboard = pd.DataFrame(scores).sort_values(["PQ", "Dice", "variant"], ascending=[False, False, True])
leaderboard.to_csv(RUN / "preprocessing_comparison.csv", index=False)
WINNER = str(leaderboard.iloc[0]["variant"])
BEST_CONFIDENCE = float(leaderboard.iloc[0]["confidence"])
print(f"Selected: {WINNER}; confidence={BEST_CONFIDENCE:.2f}")
display(leaderboard)
''')
md('''
## Training history, artifact gallery and native tensor evidence

Epoch mask mAP uses native merged targets; final PQ above uses the original
annotators. The SDK's standalone final validation is bypassed because it would
reconstruct its default polygon dataset. Best checkpoints are still selected by
native epoch mask mAP, and the best checkpoint is used for the PQ search.
''')
code('''
for variant in VARIANTS:
    run_dir = RUN / variant
    history = pd.read_csv(run_dir / "results.csv")
    history.columns = history.columns.str.strip()
    print(variant, "last recorded epoch")
    display(history.tail(1))
    audits = [native.load_json(p) for p in sorted(run_dir.glob("native_tensor_rank*.json"))]
    assert len(audits) == len(CFG["gpu_ids"]), f"Missing GPU rank evidence for {variant}"
    display(audits)
    memory = [native.load_json(p) for p in sorted(run_dir.glob("native_memory_rank*.json"))]
    assert len(memory) == len(CFG["gpu_ids"]), f"Missing peak-memory evidence for {variant}"
    display(pd.DataFrame(memory))
    for audit in audits:
        assert audit["image_tensor"][0] == CFG["batch_per_gpu"], "Smoke split did not exercise the requested batch size."

groups = {
    "Performance": ["results.png"],
    "Box metrics": ["BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png"],
    "Mask metrics": ["MaskF1_curve.png", "MaskPR_curve.png", "MaskP_curve.png", "MaskR_curve.png"],
    "Confusion matrices": ["confusion_matrix.png", "confusion_matrix_normalized.png"],
    "Training targets": ["labels.jpg", "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg"],
    "Validation": ["val_batch0_labels.jpg", "val_batch0_pred.jpg", "val_batch1_labels.jpg", "val_batch1_pred.jpg"],
}
for title, names in groups.items():
    paths = [RUN / WINNER / name for name in names if (RUN / WINNER / name).exists()]
    if not paths:
        continue
    columns = min(2, len(paths)); rows = (len(paths) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(8 * columns, 6 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, path in zip(axes.flat, paths):
        axis.imshow(plt.imread(path)); axis.set_title(path.name)
    fig.suptitle(f"{WINNER}: {title}"); fig.tight_layout(); plt.show()
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for variant in VARIANTS:
    thresholds = pd.read_csv(RUN / variant / "threshold_metrics.csv")
    axes[0].plot(thresholds["confidence"], thresholds["PQ"], marker="o", label=variant)
axes[0].set(xlabel="Confidence", ylabel="PQ", title="Original-annotator validation")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].bar(leaderboard["variant"], leaderboard["PQ"])
axes[1].set(ylabel="PQ", title="Best confidence per preprocessing variant", ylim=(0, 1))
fig.tight_layout(); fig.savefig(RUN / "preprocessing_comparison.png", dpi=150); plt.show()

# Visualize cached predictions without another GPU inference pass.
for record in manifest["val"][:3]:
    image = native.read_gray(record["path"])
    cached = native.load_json(native.prediction_path(native.load_json(jobs[WINNER]), "val", record))["instances"]
    masks = native.final_components(cached, BEST_CONFIDENCE, CFG["prediction_min_area"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis in axes:
        axis.imshow(image, cmap="gray"); axis.axis("off")
    with np.load(record["target"]) as target:
        merged = target["labels"] > 0
    axes[0].imshow(np.ma.masked_where(~merged, merged), cmap="autumn", alpha=0.6)
    axes[0].set_title("All-annotator union (visual reference)")
    for i, rle in enumerate(masks):
        mask = native.decode_mask(rle)
        axes[1].contour(mask, levels=[0.5], colors=[plt.get_cmap("tab10")(i % 10)], linewidths=0.7)
    axes[1].set_title(f"{WINNER}: {len(masks)} predicted components")
    fig.suptitle(record["name"]); fig.tight_layout(); plt.show()
''')
md('''
## Submission from the selected variant, using both GPUs

Each test source is processed exactly once. Native compressed COCO RLE is written
with columns `filament_id,segmentation_rle`, in deterministic source/component order.
An explicit coverage manifest records zero-detection sources too. Smoke mode writes
`smoke_submission.csv`; full mode writes `submission.csv`. Only validation chooses
the preprocessing variant and confidence; test images supply no selection metrics.
''')
code('''
if CREATE_SUBMISSION:
    native.launch_prediction(jobs[WINNER], RUNTIME_PATH, "test")
    submission_path = native.create_submission(jobs[WINNER], BEST_CONFIDENCE)
    print(f"Submission: {submission_path}")
    display(native.load_json(submission_path.with_suffix(".manifest.json")))
    display(pd.read_csv(submission_path).head())
''')
md('''
## Reading and extending the experiment

- `data_audit.json`, `manifest.json`, `merge_preview.png`: source grouping, split and target audit.
- `preprocessing_preview.png`, `preprocessing_comparison.csv`: image transforms and validation comparison.
- `<variant>/weights/{best,last}.pt`, `results.csv`, `logs/`: training artifacts and recovery logs.
- `<variant>/native_tensor_rank*.json`: actual native tensor shapes and GPU ranks.
- `<variant>/native_memory_rank*.json`: per-rank peak training memory and observed batch sizes.
- `<variant>/validation_per_annotator.csv`, `validation_per_source.csv`, `threshold_metrics.csv`: scoring evidence.
- `<winner>/submission.csv` and its manifest: full-run submission and source coverage.

Change `VARIANTS` to select experiments. Change preprocessing parameters, training
settings or target policy with a new `run_name`. Full and smoke directories are
separate. Keep the generated runtime alongside checkpoints when moving artifacts.
No full training score or dual-T4 performance is claimed by this notebook itself.

Implementation references: [Ultralytics configuration](https://docs.ultralytics.com/usage/cfg/)
and [pinned 8.4.198 loss implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.198/ultralytics/utils/loss.py).
''')

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
destination = HERE / "solar-filament-yolo26-native-2048.ipynb"
nbf.write(nb, destination)
print(destination)
