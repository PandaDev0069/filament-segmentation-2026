"""Build the standalone Kaggle notebook from the reviewable runtime source."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
runtime = (HERE / "pq50_runtime.py").read_text(encoding="utf8")
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value.strip()))


def code(value):
    cells.append(nbf.v4.new_code_cell(value.strip()))


md("""
# Solar filaments: native mask ensemble + instance proposals + learned PQ selection

**Target: improve on the 0.33 public run. `pq50` is an experiment name and a 0.50 goal,
not a measured score or a guarantee.** Run all cells on Kaggle **GPU T4 ×2**, with the
competition data attached and Internet enabled for packages and generic pretrained weights.

The saved YOLO11 run scored **0.4234 held-out PQ**, with **473 TP / 211 FP / 329 FN**;
public PQ was 0.33. Under-500-pixel calibration recall was only **23.4%**. Disk-fit
failures had **0.0927 report PQ**, although that cohort is small and confounded by
image difficulty. Extending the same YOLO training is not the main experiment here.

This notebook trains a **YOLO11m box detector at 2048**, a **ResNet34 U-Net on native
512 crops**, and an **EfficientNetV2-S U-Net on native 768 crops**. The mask ensemble
predicts foreground and contours on overlapping native tiles. Detection boxes keep
instance identity; semantic components can recover detector misses. A small learned
quality regressor estimates whether a mask earns official PQ credit. Calibration
compares hybrid and semantic-only output, learned and simple selection, three pixel
thresholds and two minimum sizes. An adaptive candidate mode lets learned quality
choose different mask thresholds for different instances, with nested-duplicate
suppression. It does not assume hybrid fusion is always better.

Two GPUs train with DDP; each rank gets **two loader workers**, four total. Test
inference shards images between two GPU processes. Preprocessing and candidate
evaluation use four bounded CPU workers. Exact flips/90° rotations preserve crop
detail; full disks are not resized for the mask network. Detector resizing is only
used in the explicitly reduced smoke mode.
""")

md("""
The split unit is the **physical observation date**, before annotator expansion.
Training samples one annotator view at a time and oversamples small instances.
It never makes an all-annotator hard union. All images receive photometric
normalization; uncertain solar geometry only disables radial correction.

Checkpoints are selected with a fixed-rule calibration PQ every four epochs;
postprocessing does **not** run after each training batch. Candidate-quality scores
are crossfitted by calibration date. The report set is excluded from all fitting
and recipe selection. Calibration remains a **selection score**, because it also
selects checkpoints. The independent report measures the selected pre-refit
pipeline. With `refit_full=True`, the frozen schedule then retrains on all supplied
training annotations; the earlier report score is **not** a score for that refit.

Default full mode performs selection training **and full-data refitting**. This is
a substantial training run, not a five-epoch quick fix. Runtime on two T4s has not
been benchmarked. Epoch logs show time, optimizer updates and VRAM. Completed
stages and prediction files are reused within the same fingerprinted run directory.
Use `mode="smoke"` first to test your environment; its small CSV is explicitly
named `smoke_submission.csv` and must not be submitted.
""")

code("""
from pathlib import Path
import importlib.metadata as metadata
import subprocess
import sys

# Keep Kaggle's CUDA-enabled torch/torchvision installation. Do not replace it.
requirements = {
    'ultralytics': ('ultralytics', '8.4.198'),
    'segmentation_models_pytorch': ('segmentation-models-pytorch', '0.5.0'),
    'timm': ('timm', '1.0.30'),
    'scikit-learn': ('scikit-learn', '1.9.1'),
}
install = []
for distribution, (package, version) in requirements.items():
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        installed = None
    if installed != version:
        install.append(f'{package}=={version}')
for package in ['pycocotools', 'scipy', 'opencv-python-headless', 'pandas', 'matplotlib', 'joblib', 'pyyaml']:
    try:
        metadata.version(package)
    except metadata.PackageNotFoundError:
        install.append(package)
if install:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet', *install])
print('Dependencies ready. If these libraries were imported earlier in this kernel, restart it before proceeding.')
""")

code("from pathlib import Path\nRUNTIME_PATH = Path.cwd() / 'pq50_runtime.py'\nRUNTIME_SOURCE = " + repr(runtime) + "\nRUNTIME_PATH.write_text(RUNTIME_SOURCE, encoding='utf8')\nprint('Embedded standalone runtime:', RUNTIME_PATH)")

code("""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
import importlib
import torch
import pq50_runtime as pipeline
importlib.reload(pipeline)

CFG = pipeline.default_config()
CFG.update(
    mode='full',                 # 'smoke': three sources/partition, one short training epoch
    data_root='',                # auto-discover local/Kaggle competition train/ and test/
    output_root='/kaggle/working' if Path('/kaggle/working').exists() else './pq50_outputs',
    run_name='native_hybrid_v1',
    gpu_ids=[0, 1],               # GPU T4 x2. Explicitly use [0] for a local single-GPU check.
    cpu_workers=4,               # TOTAL: two training loader workers per GPU rank
    refit_full=True,              # frozen epoch counts, fresh training on all 707 images
)
# Resource controls, if needed: reduce dense_batch_per_gpu (2 -> 1) or
# inference_batch (2 -> 1). Keep crop/tile/detector sizes fixed for comparable results.
print('CUDA GPUs:', [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])
if not torch.cuda.is_available() or max(CFG['gpu_ids']) >= torch.cuda.device_count():
    raise RuntimeError('Select Kaggle GPU T4 x2, or explicitly configure the GPUs available on your machine.')
print('Mode:', CFG['mode'], '| GPUs:', CFG['gpu_ids'], '| CPU workers total:', CFG['cpu_workers'])
""")

md("""
## Prepare shared inputs and independent annotation targets

The notebook measures compressed cache size and checks **incremental** disk needs.
Detector records hard-link the same image instead of creating full RGB copies.
Only best/last weights and compact prediction maps are retained. No automatic
recursive deletion is performed. Changing code, labels or configuration creates
a separate fingerprinted run, protecting older experiments.
""")

code("""
CFG, MANIFEST = pipeline.prepare(CFG)
RUN = Path(CFG['run'])
print('Run directory:', RUN)
for part, sources in MANIFEST['splits'].items():
    print(part, 'images=', len(sources), 'annotator records=', sum(len(s['views']) for s in sources),
          'geometry fallbacks=', sum(not s['photometry']['geometry_ok'] for s in sources))
""")

code("""
import cv2
import numpy as np
import matplotlib.pyplot as plt

examples = MANIFEST['splits']['train'][:3]
fig, axes = plt.subplots(len(examples), 3, figsize=(13, 4 * len(examples)), squeeze=False)
for row, source in zip(axes, examples):
    row[0].imshow(pipeline.read_gray(source['image']), cmap='gray')
    row[0].set_title(source['stem'])
    row[1].imshow(cv2.imread(source['cache'])[:, :, 1], cmap='gray')
    row[1].set_title('Normalized; geometry=' + str(source['photometry']['geometry_ok']))
    packed = cv2.imread(source['views'][0]['target'], 0)
    row[2].imshow((packed & 1) > 0, cmap='gray')
    row[2].set_title('One annotator foreground target')
    for axis in row:
        axis.axis('off')
fig.tight_layout()
fig.savefig(RUN / 'preprocessing_preview.png', dpi=130)
plt.show()
""")

md("""
## Train detector and native mask models

Each training job launches an importable script with `torchrun`, avoiding notebook
DDP pickling problems. Jobs train sequentially; each uses both GPUs. Mask losses
combine boundary-weighted BCE, foreground Dice and an auxiliary contour loss.
Encoder BN statistics are frozen, decoder normalization uses GroupNorm, and EMA
weights are evaluated. The cosine schedule completes its planned decay.

Full-mode detector validation uses one image and zero additional loader workers;
training uses two loader workers per rank. Generic COCO/ImageNet initialization
is downloaded, with **no weights trained on external solar-filament labels**.
""")

code("""
pipeline.train_all(CFG, full=False)
print('Selection model checkpoints:', pipeline.model_paths(CFG, full=False))
""")

code("""
import pandas as pd
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for member in range(len(CFG['dense_models'])):
    history = pd.read_json(RUN / 'selection_models' / f'dense_{member}' / 'history.jsonl', lines=True)
    label = CFG['dense_models'][member]['encoder']
    axes[0].plot(history.epoch, history.loss, label=label)
    checked = history.dropna(subset=['fixed_rule_calibration_PQ'])
    axes[1].plot(checked.epoch, checked.fixed_rule_calibration_PQ, marker='o', label=label)
axes[0].set(xlabel='Epoch', ylabel='Training loss')
axes[1].set(xlabel='Epoch', ylabel='Fixed-rule calibration PQ (selection)')
for axis in axes:
    axis.legend()
    axis.grid(alpha=.2)
fig.tight_layout()
fig.savefig(RUN / 'training_history.png', dpi=150)
plt.show()
""")

md("""
## Calibrate on dates excluded from training

Inference averages four exact flip views from both U-Nets, with Hann-weighted tile
blending. YOLO boxes are fused across views while retaining separate proposals.
Masks from proposals can retain disconnected pieces as one instance. Every final
pixel belongs to at most one mask. Plain semantic components are a competing
pipeline and an optional detector-miss recovery source.

Candidate generation retains at most 512 semantic components per threshold, ranked
by mean probability when that guard is reached. This bounds work for an untrained
or failing model while retaining small confident components. Normal filament counts
are far below this guard; `max_dense_candidates` is configurable.

The quality model uses image-derived contrast, shape, uncertainty, contour strength,
detector confidence and TTA support. It has no filename/site/label metadata at
inference. Calibration-only targets are the mean credited IoU over annotation
views. Date-grouped crossfitting prevents scoring each calibration candidate with
a quality model fitted on that date. The report will use a model fitted on all
calibration dates and the frozen recipe.
""")

code("""
CALIBRATION_MAPS = pipeline.predict_part(CFG, 'calibration', full=False)
CALIBRATION_CANDIDATES = pipeline.build_candidate_cache(CFG, 'calibration', CALIBRATION_MAPS)
RECIPE = pipeline.calibrate(CFG, CALIBRATION_CANDIDATES)
display(pd.DataFrame(pipeline.load_json(RUN / 'calibration_grid.json')).head(12))
""")

md("""
## Evaluate the frozen pipeline once on the report partition

The implementation follows the organizer's strict **IoU > 0.5** overlap rule and
pools counts across **individual annotator-image records**. It does not score a
merged union or average per-image PQ. Keep this result as the honest pre-refit
estimate. Do not tune repeatedly against this set or assume it equals public PQ.
""")

code("""
REPORT_MAPS = pipeline.predict_part(CFG, 'report', full=False)
REPORT_CANDIDATES = pipeline.build_candidate_cache(CFG, 'report', REPORT_MAPS)
REPORT = pipeline.evaluate_report(CFG, REPORT_CANDIDATES, RECIPE)
display(pd.DataFrame([REPORT]))
pipeline.preview(CFG, 'report', REPORT_CANDIDATES, RECIPE)
plt.show()
""")

md("""
## Frozen full-data refit and submission.csv

When enabled, retrain on train + calibration + report annotations using the epoch
counts already selected during the earlier stage. Keep the quality model and
postprocessing frozen. This uses all legal training data but has no remaining
independent local score; the saved report belongs to the previous models.
If semantic-only masks win calibration, skip the unused detector's full-data retrain.

The final CSV uses `filament_id,segmentation_rle`, native 2048² column-major COCO
compressed RLE counts, original filename stems and one-based instance counters.
Empty images are logged without invented masks. The exporter checks identifiers,
all test-image coverage, mask dimensions, RLE round trips and pixel non-overlap.
""")

code("""
if CFG['refit_full']:
    pipeline.train_all(CFG, full=True)
TEST_MAPS = pipeline.predict_part(CFG, 'test', full=CFG['refit_full'])
TEST_CANDIDATES = pipeline.build_candidate_cache(CFG, 'test', TEST_MAPS)
SUBMISSION = pipeline.export_submission(CFG, TEST_CANDIDATES, RECIPE)
print('Submission:', SUBMISSION)
display(pd.read_csv(SUBMISSION).head())
display(pipeline.load_json(RUN / 'submission_audit.json'))
""")

code("""
pipeline.preview(CFG, 'test', TEST_CANDIDATES, RECIPE)
plt.show()
from IPython.display import FileLink, display
display(FileLink(str(SUBMISSION)))
print('Full runs also copy submission.csv to:', Path(CFG['output_root']).resolve() / 'submission.csv')
print('Review report_metrics.json, submission_audit.json and the previews before uploading.')
""")

md("""
## Research, rules and limits

The quantitative metric changed to PQ in August 2026. Old Dice leaderboard numbers
are not comparable. The organizer permits computational approaches but prohibits
other ground-truth metadata for training. This notebook uses only the competition
training annotations and generic COCO/ImageNet initialization. It does not retrieve
the public MAGFiLO annotation release, which overlaps competition test observations.

Primary sources reviewed:

- [Competition evaluation and data restrictions](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates)
- [Organizer self-evaluation](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook)
- [Amey Pawar's detector/native-U-Net hybrid](https://github.com/ameyypawar/filament-seg): author-reported 0.4214 crossfit / 0.37 public; supports complementary proposals and masks, not a 0.50 promise.
- [HeShen's detection/refinement studies](https://github.com/HeShen-1/filament-segmentation-2026): author-reported 0.3593 five-fold / 0.35 public; documents correlated fusion errors and non-transferring training-length gains. Human agreement is not a mathematical model ceiling.
- [Artem Shelamanov's pipeline account](https://www.shelamanov.com/portfolio/solar): author-reported 0.40 public with native ROI masks and learned quality/fragment handling; code was unavailable when reviewed.
- [SMP U-Net documentation](https://smp.readthedocs.io/en/latest/models.html), [timm encoder interface](https://smp.readthedocs.io/en/latest/encoders_timm.html)
- [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [Ultralytics training](https://docs.ultralytics.com/modes/train/)

This is a new implementation informed by those designs, not downloaded competition
weights or copied submissions. Dependencies retain their own licenses, including
Ultralytics AGPL-3.0. No independently reproduced rule-compliant 0.50 public recipe
was established by this research. Mask/proposal fusion and full-data refitting may
help or hurt; the notebook records evidence rather than manufacturing a score.

The accompanying workspace research folder contains the full training-label audit
and independent analysis of `Best/results.zip`. Smoke checks validate mechanics;
they do not establish model quality or validate two-T4 performance on a one-GPU PC.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}})
target = HERE / "solar-filament-native-hybrid-pq50.ipynb"
nbf.validate(nb)
nbf.write(nb, target)
print(target)

