# Native 2048 × 2048 YOLO26 pipeline

Open [solar-filament-yolo26-native-2048.ipynb](solar-filament-yolo26-native-2048.ipynb).
The saved best-run notebook and `results/` are preserved. The new notebook contains
the complete implementation and automatically writes its runtime module, so it can
be uploaded to Kaggle alone.

Select **GPU T4 ×2**, attach the dataset, and run the default `MODE = "smoke"`.
Inspect the native tensor audits, annotation union and preprocessing previews.
Then use `MODE = "full"` for the 150-epoch experiments and competition submission.
All settings are in the first code cell. `VARIANTS` selects raw, CLAHE, percentile
contrast and gamma experiments. Each trains independently on the same split.

The pipeline uses YOLO26s segmentation. Inputs and exact training target masks
remain 2048 × 2048. Incorrect dimensions fail immediately. The dataset loader
bypasses the SDK's resizing, letterboxing, polygon conversion and augmentation
pipeline. Exact pixel flips and brightness augmentation remain enabled; rotation,
scaling, crops, tiling and mosaic are disabled. Image preprocessing changes
intensities only and is identical during training, validation and test.

All annotator records for a source filename are combined with pixelwise OR.
8-connected components become native instance-ID maps. One-pixel components and
thin lines survive, and holes remain intact. Touching filaments can merge under
this policy. `merge.min_area` and `prediction_min_area` are separate settings,
both defaulting to 1. The previous run used a component-area filter of 20.

The native mask loss upsamples predicted mask logits against unchanged native
targets. It uses checkpointed per-anchor loss computation to limit memory, and
standard box assignment to avoid the stock YOLO26 mask assigner's target
downsampling. Internal encoder features and mask prototypes still have lower
spatial resolution. This changes training and should be assessed by validation;
no improved competition score is claimed.

Original annotator masks remain the validation reference. Each source's score is
averaged across its annotators, then sources receive equal weight. PQ uses strict
IoU > 0.5 with one-to-one matching. Source filenames and pixel-identical images
cannot cross the split. The split and metric averaging differ from the saved
baseline, so compare variants within this experiment rather than interpreting its
scores as an exact replication of the old validation run.

| Stage | GPU scheduling | CPU work |
|---|---|---|
| Source audit and native target preparation | CPU | Four bounded threads |
| Training each variant | Both GPUs with DDP; two images per GPU (global batch 4) | Two loader workers per rank; validation loaders have zero workers |
| Validation and test prediction | One replica per GPU, disjoint source shards | Two threads per GPU for prefetch and RLE encoding |
| Threshold selection and submission | Cached native masks on CPU | Four bounded threads and one ordered CSV writer |

Variants run sequentially, with both GPUs used by each. GPU memories remain
separate. The allocator cap is 13 GiB per T4, limited further to 90% of the actual
device's VRAM. There is no automatic reduction of image size. Image caching is
disabled, and variants share exact target files rather than materialized image
copies. Lower batch size, model size or loader workers if required by a different
hardware profile.

The default smoke split now has eight sources per partition, giving each GPU two
full batches of two images. The earlier two-source smoke used only one image per
rank and could not measure batch-two memory. For such tiny splits the SDK disables
extra loader workers; `Using 0 dataloader workers` refers to CPU loading, not GPU use.
The configured training loader budget remains two workers per rank. Weights are
resolved/downloaded in the coordinator before DDP starts, preventing two ranks
from writing the same download. Inference uses `quantize=16` for FP16, and unused
Ray Tune integration is disabled.

Each rank writes `native_memory_rank*.json` with actual batch sizes and peak
allocated/reserved training memory. Two images per GPU is the new starting setting,
based on the supplied one-image T4 log reporting 5.26 GiB; the larger batch has not
yet been measured on T4s. Check both ranks' reports before trying three images per
GPU. Native validation and prediction keep one image per GPU at a time.
Peak reports use [PyTorch's peak memory counters](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html),
reset for each training epoch. The default run name is now
`yolo26s_native_union_b2`, keeping these results separate from batch-one runs.

Each variant writes checkpoints, histories, logs, native tensor evidence,
per-annotator/source validation metrics and threshold trials. The highest
validation-PQ variant generates the test submission. Full mode writes
`<run>_full/<winner>/submission.csv`; smoke mode writes `smoke_submission.csv`.
The accompanying manifest records all test sources, including zero detections.

Use a new `run_name` when changing experiment settings. Unchanged completed
training jobs are reused. For an interrupted run, set `resume=True` to recover
`last.pt`; the contract rejects incompatible settings. Moving custom checkpoints
also requires the generated `native_yolo_runtime.py`; rerunning the runtime cell
recreates it. Kaggle's existing CUDA PyTorch/torchvision should be retained.
The notebook requires `ultralytics==8.4.198` and installs missing auxiliary packages.
When replacing an older notebook/runtime revision, choose a new `run_name` for an
existing mode. Switching from smoke to full already creates a separate directory.

For development, [native_yolo_runtime.py](native_yolo_runtime.py) is the source of
the embedded module. After editing it, rebuild the notebook:

```powershell
.\.venv\Scripts\python Best\build_native_notebook.py
.\.venv\Scripts\python -m pytest Best\test_native_yolo.py -q
```

See [VALIDATION-native-2048.md](VALIDATION-native-2048.md) for measured execution
evidence and the remaining dual-GPU hardware check.
