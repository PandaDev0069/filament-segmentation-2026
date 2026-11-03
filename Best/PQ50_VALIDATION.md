# Native hybrid notebook validation

Validated on 3 November 2026 in `D:\filament-segmentation-2026`.

**The new notebook is implemented and mechanically tested. A >0.50 score is not
established. Full competition training and a Kaggle submission have not been run.**

Use [solar-filament-native-hybrid-pq50.ipynb](solar-filament-native-hybrid-pq50.ipynb).
It embeds [pq50_runtime.py](pq50_runtime.py) and needs no companion code upload.
The previous preprocessing notebook was preserved.

The [research/design notes](../research/filament_competition_2026-11-03/PQ50_NOTEBOOK.md)
explain the evidence, restrictions, method and remaining uncertainties.

## Checks completed

| Check | Result |
|---|---|
| Notebook JSON schema, code-cell compilation, embedded/runtime equality | Passed |
| Behavioral test suite | **33 passed** |
| Actual-model two-rank Gloo CPU DDP, two training updates | Passed; identical final parameter hashes |
| Pretrained ResNet34 U-Net, native 512 crop, batch 2, CUDA AMP backward + AdamW update | Passed; 0.814 GiB peak allocated |
| Pretrained EfficientNetV2-S U-Net, native 768 crop, batch 2, CUDA AMP backward + AdamW update | Passed; 2.888 GiB peak allocated |
| All trainable parameters receive finite gradients in the native CUDA checks | Passed |
| Staged training → native inference → candidate scoring → calibration → report → CSV smoke flow | Passed |
| Smoke CSV dimensions, unique IDs, RLE decode/re-encode equality and no overlaps | Passed for 67 masks across all three smoke test images |
| Smoke preview rendering | Completed and visually inspected |
| Full dataset indexing | 707 physical training images, 1,154 views, 8,199 masks; 180 test images |

The CUDA checks ran on a single RTX 3070 Ti Laptop GPU, 8 GiB, with Python 3.12.10,
PyTorch 2.14.1+cu132, SMP 0.5.0, timm 1.0.30, Ultralytics 8.4.198 and scikit-learn
1.9.1. Peak allocated memory above covers a model, training batch, gradients and
AdamW step; it is not a measured total for a complete T4 training session with EMA,
data workers, caching and CUDA/NCCL context. An exploratory batch-three EfficientNet
check was stopped after severe slowdown on the shared 8 GiB Windows GPU. Both
production models subsequently passed at the final default batch of two.

The DDP test found an actual issue before delivery: SMP's final decoder block
contains SCSE parameters for an absent skip connection. Those dormant parameters
are now frozen, preventing a second-iteration DDP reduction failure. The corrected
two-rank model test passes. **Actual dual-T4 CUDA/NCCL execution and full-run
runtime have not been tested locally**, because only one CUDA GPU is available.

The behavioral suite covers official strict `IoU > 0.5` and all-hit semantics,
empty predictions/targets, pooled annotator counts, independent annotator quality
targets, physical-date grouping, content fingerprints, rerun idempotency,
small/empty-crop loss gradients, coordinate-preserving tile/D4 inverses, sparse
ROI RLE equality against native COCO encoding, exclusive pixel ownership,
fragment identity, adaptive-threshold nested-duplicate suppression, date-crossfitted
quality scoring, semantic-only refit handling, test coverage and CSV export.

The short end-to-end run used three physical sources per partition, a one-epoch
YOLO11n detector and a randomly initialized ResNet18 mask model with **two optimizer
updates**. Its calibration/report PQ is zero and its masks are poor. This is an
execution check, **not a quality estimate for the full pretrained production
pipeline**. The staged smoke reused its completed trained weights while testing
the final calibration/export code. It is not a claim that the full notebook was
executed unchanged on two T4s. Its deliberately non-submittable artifact is
`pq50_validation/end_to_end_3997afa6f7/smoke_submission.csv`, SHA256
`b6173c6303a935a6f57216310166e859c37c8f3f7d7e5362e83880c0ba6fe2d4`.

## Full-run defaults

The date-grouped split contains **532 train / 82 calibration / 93 report** physical
images, corresponding to **867 / 134 / 153** annotator views and **6,121 / 999 /
1,079** masks. Test has 180 images. Each original annotation record has at most
26 instances in this training release; the configurable 512-component generation
guard is much larger and is primarily protection against noisy/untrained models.

Two GPUs use DDP with **two data-loader workers per rank, four total**. Dense
training uses batch two per rank, accumulation three, 288 microbatches per epoch
and 32 planned epochs. The default effective global batch is 12, with 96 planned
optimizer updates per epoch. Refit scales crop steps with the larger training population,
preserving per-image exposure, while keeping selected epoch counts frozen.
The detector uses YOLO11m at native 2048, batch one per GPU, for 45 planned epochs.
Masks infer on overlapping native 1024 tiles with four flip views and both encoders.

The notebook compares fixed probability thresholds and learned per-instance
threshold hypotheses. Calibration can choose semantic-only output; when it does,
the unused detector's full-data refit is skipped. Report and test masks remain
non-overlapping regardless of the selected family.

Set `mode='smoke'` for an environment check, then `mode='full'` for real training.
Full mode is already the delivered default. Attach competition data and enable
Internet for package and generic pretrained-weight acquisition. The final full
run generates **`/kaggle/working/submission.csv`**, plus the frozen recipe,
calibration grid, independent pre-refit report, predictions, training histories,
model provenance, previews and `submission_audit.json` in its run directory.

No public/test ground truth, external filament annotations or competition-trained
third-party checkpoints were used. The full run remains necessary to establish
whether this approach improves actual PQ and whether it reaches the requested goal.

Reproduce local checks:

```powershell
.\.venv\Scripts\python.exe Best\build_pq50_notebook.py
.\.venv\Scripts\python.exe -m pytest Best\test_pq50_pipeline.py -q
.\.venv\Scripts\python.exe Best\verify_pq50_models.py --task ddp
.\.venv\Scripts\python.exe Best\verify_pq50_models.py --task cuda --pretrained --batch 2
```

The unit suite emits pycocotools/NumPy deprecation warnings; they do not affect the
verified RLE equality. Large local smoke artifacts and generic downloaded weights
are excluded from Git under `pq50_validation/.gitignore`.
