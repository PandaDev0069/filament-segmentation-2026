# November 4 notebook failure and correction

Use **[solar-filament-kaggle-fixed.ipynb](../solar-filament-kaggle-fixed.ipynb)**.
It is a complete, independent Kaggle notebook. Attach the competition data, select
T4 x2, enable Internet and run it in a fresh session. Its default input/model
requirements are unchanged: supplied competition data and automatically downloaded
generic pretrained models. The executed notebook and original logs are preserved.

## What the run actually shows

The executed helper source matches the previously delivered standalone notebook.
`log.txt` and `solar-filament-notebook.log` are identical copies. The text log
starts late in the run; the notebook's captured output contains the earlier
training history and reveals that failure started before the final refit.

| Stage | Completed epochs | First NaN-loss epoch | Epochs with NaN loss |
|---|---:|---:|---:|
| Initial ResNet34 mask training | 32 | 3 | 30 |
| Initial EfficientNetV2-S mask training | 32 | 2 | 31 |
| Full-data ResNet34 refit | 4 | None logged | 0 |
| Full-data EfficientNetV2-S refit | 12 | 2 | 11 |

ResNet34's fixed-rule calibration PQ fell from 0.14815 at epoch 4 to zero by
epoch 12. EfficientNetV2-S's fixed-rule scores stayed near 0.02. The selected final
calibration recipe scored **0.01117 PQ**, with 9 TP and 990 FN. The independent
report scored **0.003451 PQ**, with **3 TP, 17 FP and 1,076 FN**.

The output then emitted `RuntimeWarning: invalid value encountered in cast` at
the conversion of floating mask probabilities to uint8. It nevertheless finished
with **six masks across all 180 test images**. The GPUs were both used: the logs
show two ranks, two loader workers per rank, and 90 test images per GPU. The
conversion/nbconvert syntax warnings at the end are unrelated to the model failure.

This was a numerical and validation failure in the supplied implementation.
Passing CSV format checks did not make the underlying predictions valid.

## Failure path in the previous code

The mask model forward ran under FP16 autocast. Casting its output to float32
inside BCE/Dice or before sigmoid cannot repair a NaN/Inf already produced by
the forward pass. No finite checks stopped training, rejected contaminated
checkpoints or validated each model's output before averaging. The probability
encoder then cast invalid values to integers, losing those mask probabilities.

The logs do not identify the first failing neural-network operation or record
gradient-scale values. Therefore FP16 overflow at a particular layer, checkpoint
parameter corruption and scale underflow cannot each be proven from these files.
The observed non-finite losses and inference conversion warnings are conclusive.
Using float32 throughout the mask path is the conservative numerical correction.
PyTorch explicitly notes that AMP/FP16 is not compatible with every model and
that gradient scaling does not guarantee a scale above one.
[PyTorch AMP documentation](https://docs.pytorch.org/docs/2.15/amp.html#gradient-scaling)

The old update counter also inferred success from `new_scale >= old_scale`.
If both scales were zero, that could count a skipped step as an update. The
logs do not contain scales to establish whether this happened. The corrected
mask path counts direct optimizer steps and does not use GradScaler.

The pipeline always averaged both mask networks, even though one could be much
worse. Calibration then selected a filter from already poor predictions. Full-data
refitting replaced the evaluated models without an independent score for the new
weights and introduced another long NaN-loss stage. There was no collapse check
to stop these later GPU jobs or the six-mask submission.

## Corrections implemented

- Mask forward, backward, validation and inference use **float32**. YOLO retains
  its independently managed AMP; the logged detector training was not the source
  of the mask-loss failure.
- The pretrained encoder learns at **0.1 times the decoder learning rate** to
  reduce abrupt changes while its BatchNorm statistics are frozen.
- Inputs, targets, logits, loss, gradient norm and updated model parameters are
  checked for finite values. Training ranks synchronize failure status so one
  bad rank cannot leave another waiting in backward. Updates and EMA changes
  proceed only after the relevant checks succeed.
- Checkpoint saves and loads reject non-finite tensor state and metadata. History
  JSON forbids NaN values. The first epoch is evaluated, followed by regular checks,
  so a healthy early checkpoint need not wait until epoch four.
- Every native inference window is checked **before sigmoid**. Probabilities are
  checked again before averaging and quantization. NaN/Inf conversion raises an
  explicit error instead of generating a seemingly valid sparse CSV.
- Calibration compares **ResNet34, EfficientNetV2-S and their average**. The chosen
  mask source is used consistently for calibration candidates, report and test;
  unselected networks are excluded from the submission path.
- **Full-data refitting is disabled by default.** Test uses the models whose
  report was actually evaluated. Optional refitting remains available after a
  valid report and uses only the selected mask members.
- A full-run report below **0.10 PQ** stops refitting/export as an explicit
  collapse check. This threshold would have stopped the failed run. It is not a
  claim that 0.10 is competitive or that 0.50 is achieved. Smoke mode bypasses
  this quality gate because its models receive only a few training steps.
- The runtime version, run directory prefix, prediction schema and candidate
  cache profile changed. Old zeroed probability caches are not reused by the
  corrected default workflow.

The notebook remains self-contained and uses two GPUs with four total loader
workers. No additional labeled data, test ground truth or manually uploaded
weights have been introduced.

## Verification and limits

The regression suite covers the actual failure cases: non-finite native logits,
loss/gradient rejection, checkpoint poisoning, NaN-to-uint8 rejection, model
selection excluding a weak member, and stopping a collapsed report before export.
It also retains the prior label, official PQ, ownership, RLE and notebook-isolation
checks. Exact current test counts and sustained results are recorded in the final
validation section below.

A two-process CPU DDP test completed actual training updates with identical final
weights, then injected a NaN on one rank and verified that both ranks stopped
together. This exercises failure coordination, not dual-T4 CUDA/NCCL hardware.

Sustained GPU checks use real competition training images and original target
masks, native 512/768 crops, pretrained encoders, float32, real optimizer/EMA
updates, full 2048 inference with 1024 tiles and D4 views, strict quantization and
checkpoint validation. They use one local RTX 3070 Ti Laptop GPU and a small
training subset. These checks test numerical behavior and learning; they are not
a full validation benchmark or evidence of a particular public leaderboard score.

The failed Kaggle run used Python 3.13.16 / PyTorch 2.11.0+cu130 on two T4s. Local
verification uses Python 3.12.10 / PyTorch 2.14.1+cu132 on one RTX 3070 Ti. A full
corrected Kaggle run is still needed to establish model quality. No >0.50 result
is claimed.

Forensic captures are `run_summary.json`, per-cell output text and
`executed_worker_source.py`. The latter is the **old failing source for inspection**,
not the corrected worker to run. Local test-only weights and results are under
`validation/` and are excluded from Git.


## Final validation results

**56 automated checks pass.** The two-rank CPU test also confirms identical
weights after real updates and synchronized rejection of a NaN on one rank.

Sustained real-image GPU tests, batch one and float32:

| Encoder | Native crop | Actual optimizer updates | First 32 mean loss | Last 32 mean loss | Native 2048 prediction / quantization / checkpoint |
|---|---:|---:|---:|---:|---|
| resnet34 | 512 | 256 | 0.7809 | 0.3307 | Passed |
| tu-tf_efficientnetv2_s | 768 | 256 | 0.9740 | 0.2491 | Passed |

Both networks processed foreground-containing crops, rather than only empty
backgrounds (189/256 ResNet34 crops; 229/256 EfficientNetV2-S crops). Each test
then ran a full native 2048 image using overlapping 1024 tiles and four flip views.
All resulting floating probabilities and checkpoint tensors were finite.

The production batch-two FP32 forward/backward/optimizer checks also passed.
Peak allocated CUDA memory was 0.988 GiB for ResNet34 and 4.569 GiB for EfficientNetV2-S.
These are measured local allocated-memory peaks, not a complete two-T4 runtime
or total system-memory benchmark.

The actual training-to-CSV subprocess smoke test **completed successfully**:
detector training, mask-worker training, native prediction, model selection,
quality fitting, calibration, report, test prediction and CSV export. It wrote
69 masks across all four smoke test images; native dimensions, unique IDs,
compressed RLE round trips and pixel non-overlap all passed. The status is
recorded in `validation/subprocess_smoke.json`. The smoke
run uses four sources per partition, one short detector epoch and two mask
updates; its predictions are not a model-quality benchmark or a submission.


`final_validation.json` records these results, the numerically tested notebook
SHA256, a subsequent visualization update with its own SHA256/checks, and
confirmation that the original executed notebook and both logs are unchanged.

The visualization update enables YOLO training artifacts and adds grouped image
display, YOLO CSV histories and six U-Net history panels. It was rendered against
saved smoke-run logs and checked with ten relevant notebook packaging/compilation
checks. This display change does not alter the repaired numerical training logic.
