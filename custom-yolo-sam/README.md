# Native-resolution solar filament YOLO + SAM

Open [solar_filament_yolo_sam_native.ipynb](solar_filament_yolo_sam_native.ipynb).
It contains the entire pipeline, including configuration, preparation, both training
stages, validation, threshold selection, visual review, and submission generation.
No custom Python modules or generated Python worker scripts are required.
The notebook creates JSON runtime bundles and starts fresh GPU interpreters.

The pipeline combines a custom **YOLO26 detector** with a **SAM 2.1 mask decoder**
fine-tuned on solar filaments. Both models receive the original **2048 × 2048**
image. Image crops, tiles, image resizing, mosaic and spatial scaling are disabled.
SAM's feature maps have lower resolution internally; its mask logits are upsampled
to supervise against unchanged native masks.

## Run it

1. Use Python 3.12 and a compatible CUDA PyTorch/torchvision installation. On
   Kaggle, attach the competition dataset and select **GPU T4 x2**. Keep Kaggle's existing
   CUDA PyTorch build. Restart the kernel after selecting the accelerator. Enable
   Internet for the first package/weight download.
2. Edit the first code cell: `MODE`, `CFG['data_root']`, output location and model
   settings. An empty data root triggers local/Kaggle discovery.
3. Run all cells with the default `MODE = 'smoke'`. This trains on two native
   sources per partition for one epoch per model, then processes two test images.
4. Inspect `merge_preview.png`, losses and validation overlays. Set `MODE = 'full'`
   to train on the complete split and generate the competition submission.

Smoke and full runs use separate directories. The smoke CSV is named
`smoke_submission.csv`; it is an execution artifact. Full mode writes
`submission.csv`. A one-epoch detector may produce no masks; source coverage and
zero counts are still recorded explicitly.

## Kaggle resource profile

The default profile uses **two T4 GPUs with 15 GiB each and four CPU workers total**:

| Phase | GPU use | CPU worker use |
|---|---|---|
| Indexing, annotation merging, detector export | CPU preparation | Bounded four-thread pool |
| Training | YOLO on GPU 0 and SAM on GPU 1 concurrently | Two loader workers per model |
| Validation/test inference | One full YOLO + SAM replica per GPU, disjoint source shards | Two threads per GPU overlap native-image prefetch and cache writing |
| Threshold search and submission encoding | Cached predictions on CPU | Bounded four-thread pool; one ordered CSV writer |

Each GPU has a configurable **13 GiB PyTorch allocator cap**, leaving space within
its 15 GiB VRAM for CUDA. GPU memories remain separate. The default detector batch
is 2; SAM decodes 2 prompts at a time with 4-image gradient accumulation. SAM uses
one persistent loader pool across train/validation; YOLO creates no additional
validation loader workers. Tiny smoke datasets can use fewer workers. Numerical library threads and SDK scanning pools are
capped to avoid multiplying worker counts.

YOLO uses your requested AdamW, cosine decay, deterministic training, 0.5 flips,
`hsv_v=0.3`, pretrained weights, plots and verbose logs. Geometric transforms stay
disabled. Immediately after training, the notebook displays metric tables and the
requested grouped artifact gallery. Combined validation then displays SAM mask
quality as PQ, foreground Dice and IoU. Missing YOLO `Mask*` curves are explained:
this pipeline trains a box detector and a separate SAM decoder.

Configure `CFG['resources']`, `CFG['yolo']`, `CFG['sam']`, and the gallery layouts in
the first code cell. Use a fresh default `solar_yolo_sam_t4x2` run name for this
profile. Selecting `gpu_ids=[0]` makes the two training jobs run sequentially.
The Kaggle profile requires Linux. It has **not been executed on dual T4s here**;
CPU scheduling, contracts, mock subprocess lifecycle and gallery checks are recorded
separately from the previous single-GPU smoke evidence.

The notebook installs missing auxiliary dependencies and requires
`ultralytics==8.4.198`. If another version is installed, install that exact version
and restart the kernel. A CUDA NMS preflight also checks torchvision: a CPU-only
torchvision build must be replaced with a CUDA build matching PyTorch's wheel
index. For offline use, install dependencies beforehand, attach
`yolo26n.pt` and `sam2.1_t.pt`, and set their paths in the configuration.

Expected dataset:

```text
MAGFilo_1.0_Kaggle_2026/
  train/
    train_images/*.jpeg
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/
    test_images/*.jpeg
```

## Annotation merging

All annotator records with the same source filename contribute to one pixelwise
union. Connected components of that union become training masks and detector
boxes. The defaults use 8-connectivity and preserve components as small as one
pixel. There is no dilation, polygon approximation or mask resizing.

This policy can join touching filaments and split disconnected annotations. It
implements foreground union; it does not infer cross-annotator instance identity.
Original annotator masks remain intact for validation PQ. Pixel-identical source
images are grouped during splitting to prevent leakage. Multi-annotator sources
stay in training by default, following the existing U-Net split policy.

## Settings and outputs

All hyperparameters are in `CFG`, grouped by `split`, `merge`, `yolo`, `sam`,
`predict` and `search`. Change `run_name` for a new experiment. Settings incompatible
with existing training artifacts are rejected. Stage flags reuse checkpoints;
per-model `resume` settings recover interrupted training. Per-job logs appear in
`logs/`; notebook-derived runtime bundles appear in `runtime/`. The notebook prints
periodic job status and stops peer jobs when a GPU worker fails.

Outputs include the split and data audit, merged-mask preview, detector and SAM
checkpoints, epoch histories, native SAM probe, validation PQ and foreground
metrics, threshold trials, review figures, prediction caches, resumable test
journals, and a CSV with a provenance/source-coverage manifest.

See [PIPELINE.md](PIPELINE.md) for architecture, tuning, scoring and recovery, and
[VALIDATION.md](VALIDATION.md) for the actual checks performed. The current dual-T4 profile has not been hardware-tested and includes no full
competition training run or claimed competitive score.
