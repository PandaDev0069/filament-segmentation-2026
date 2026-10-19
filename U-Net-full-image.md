# Native-resolution solar filament experiment

Open `solar-filament-u-net-full-image.ipynb` and run its cells on Kaggle/Linux
with both GPUs enabled. The notebook embeds its worker and can be uploaded by
itself. `solar-filament-u-net.ipynb` remains the tiled baseline.

| Setting | Full-image experiment |
| --- | --- |
| Training and prediction input | Native 2048 × 2048 grayscale, one complete image |
| Encoder | Seven downsampling stages, 2048 → 16 |
| Channels | 16, 32, 64, 128, 256, 256, 256; bottleneck 256 |
| Detail and context | Full-resolution skips, learned upsampling, dilated and pooled bottleneck context |
| Memory | FP16 AMP, checkpointed encoder and complete decoder stages |
| GPU training | One DDP process per GPU; memories remain separate |
| Batch | Probe 1–4 images per GPU with real backward and Adam steps; choose a common fit |
| Accumulation | Four microbatches; effective batch eight at batch one on two GPUs |
| CPU/RAM | Four total training workers, prefetch one, one CPU thread per process, streamed uint8 pairs |
| Prediction | One full-image forward, then threshold and complete-image components |
| Small predictions | Minimum component area one pixel; configurable using validation |

Thin and narrow structures retain their native pixels throughout the data
pipeline. Flips and 90° rotations avoid interpolation. The final decoder uses
the full-resolution encoder features. No morphological closing is imposed.
Increasing model depth expands context but does not guarantee connected output;
the existing union-PQ/Dice/BCE loss does not explicitly enforce topology.

The split preserves the baseline's label policy: multiply annotated source files
remain in training, with each annotation record used separately. Local inspection
found 1,013 training records from 566 source files and 141 validation images,
with no source-file overlap. Validation threshold selection now operates on
complete images. Its connected-component PQ is a proxy for polygon-instance PQ
because touching instances merge in union masks.

The probe leaves a reserved-memory margin and restores the starting weights and
random state. Training writes actual per-GPU peaks, image throughput, architecture
settings and threshold results into `unet_full_image_outputs/`. A batch probe
cannot certify all future steps; inspect the actual target-device report.
Keep image resolution at 2048 if memory is tight: first fix batch at one, then
reduce channel width. If memory permits, compare width 24 or 32 using validation.

## Validation performed

All 17 existing and new regression tests pass, including real two-process
CPU/Gloo accumulation, checkpoint output/gradient equivalence, and one full-frame
inference call preserving a one-pixel filament across the former tile boundaries.
That last test validates the pipeline using a controlled model, not learned quality.

A real CUDA smoke run completed one epoch on two native-resolution training images
and one validation image, with four loader workers, followed by checkpoint reload
and a native 2048 prediction. Device: NVIDIA RTX 3070 Ti Laptop, 8 GiB, PyTorch
2.14.1+cu132. Batch one, width 16, depth seven used **3.80 GiB peak allocated /
4.26 GiB peak reserved** during the epoch; the startup probe measured 3.84 / 4.27 GiB.
See `tests/full_image_gpu_smoke_result.json`. This establishes a working full-image
path; it does not establish training quality or dual-14-GiB performance.

A user-reported run on two 14.56-GiB T4s passed batch two per GPU and reached
epoch 1 training, giving a nominal effective batch of 16 with accumulation four.
Batch three exceeded the 85% reserved-memory budget. The updated worker preserves
exact gradient strides and DDP bucket views to address the reported channels-last
1 x 1 kernel warning. Existing training can continue with the warning; the fix
applies when the updated notebook launches a new worker. Full details are in
`solar-filament-u-net-full-image-documentation.md`, sections 10 and 17.3.

Commands from the project root:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe tests/run_full_image_gpu_smoke.py
python build_full_image_notebook.py
```

The manual CUDA smoke requires the local dataset and overwrites its result JSON.
The builder requires `nbformat`; rerun it after changing the readable runtime so
the notebook's embedded worker stays synchronized.

Primary documentation: [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html),
[activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html),
[AMP and accumulation](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[GroupNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html).
