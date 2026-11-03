# New native hybrid notebook: design and evidence

Prepared 3 November 2026. Notebook: `Best/solar-filament-native-hybrid-pq50.ipynb`.
The name denotes the requested 0.50 goal. No 0.50 validation or leaderboard result
has been established for this new pipeline. The original preprocessing notebook
and its saved results are preserved.

The earlier [competition research](RESEARCH_REPORT.md) and
[independent archive analysis](latest_run/RUN_ANALYSIS.md) are the evidence base.
The actual archive is a 41-epoch YOLO11s run with 0.4234 independent report PQ,
not an unchanged five-epoch run. The user's public result is 0.33. A model-family
ceiling cannot be inferred from those observations.

## What reaching 0.50 would require

Current report SQ is 0.6651 and RQ is 0.6366. If SQ stays unchanged, 0.50 PQ
requires RQ approximately **0.7518**. With 802 annotated instances and 211 false
positives held constant, that would require roughly **610 true positives**, versus
473 currently: about **137 additional accepted matches**. These are pooled
annotator-record counts, not counts of distinct physical masks. Even eliminating
all 211 false positives while retaining the same matched masks and 329 misses
would produce only about **0.4935 PQ**. Precision filtering alone is insufficient
under that fixed-shape/count calculation.

This motivates better native masks and missed-instance recovery. It does not prove
that this implementation can supply those additional matches. The observed local
to public gap makes a 0.50 public result more demanding still.

## Research conclusions

The [organizer's evaluation](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates)
uses PQ, with strict IoU above 0.5. The metric changed in August; older Dice scores
cannot establish a successful modern recipe. Its restrictions exclude other
ground-truth metadata for training and require supplied H-alpha images at inference.
This notebook uses the supplied training annotations plus generic COCO/ImageNet
initialization; no external solar-filament labels or test pseudo-ground-truth are
retrieved. In particular, the original public MAGFiLO label release is excluded.

Public primary sources support architectural directions, not the requested result:

| Source | Author-reported evidence | Consequence for the new design |
|---|---|---|
| [Pawar's hybrid](https://github.com/ameyypawar/filament-seg) | Native U-Net + YOLO ownership; 0.4214 crossfit, 0.37 public | Learn detailed pixels separately from detection/grouping |
| [HeShen's experiments](https://github.com/HeShen-1/filament-segmentation-2026) | 0.3593 five-fold, 0.35 public; more training and correlated fusion often fail | Keep honest grouped evaluation and a semantic-only competitor |
| [Shelamanov's account](https://www.shelamanov.com/portfolio/solar) | 0.40 public; native ROI masks, annotator-aware supervision, learned quality | Evaluate candidate quality from appearance, geometry and model support |
| Supplied ten-model U-Net notebook | 0.4502 nested OOF / 0.39 public, author report | Diverse pretrained encoders and native detail are plausible, not sufficient proof |

These trained results were not independently reproduced here. Shelamanov's code
was not public when reviewed. No verified, reproducible rule-compliant >0.50 public
recipe was found. This implementation is original code using documented libraries,
not a bundle of unknown competition weights or copied predictions.

## What the new notebook changes

1. **Different representation:** YOLO11m predicts native 2048 boxes; ResNet34 and
   EfficientNetV2-S U-Nets predict native foreground/contour probabilities. The
   learned mask decoder no longer depends on YOLO's low-resolution mask prototypes.
2. **Annotator-aware labels:** each physical-image crop samples one original
   annotator record. Detector labels also keep records separate. Dates are split
   before expanding to annotators. Original masks are rasterized with COCO tools.
3. **Small-instance exposure:** 65% of crops select a training instance, with modest
   inverse-size sampling. Dark-region and random backgrounds remain supervised.
   Flips/90-degree rotations preserve native detail; contrast/brightness vary mildly.
4. **Consistent image inputs:** all images are standardized, even when geometry is
   uncertain. Only the bounded radial correction depends on a valid disk fit.
   Inputs include normalized, radially corrected and CLAHE views; none is hard-clipped
   to a guessed limb. The new transform is used for training and inference together.
5. **Stable optimization:** generic pretrained encoders, frozen encoder BN statistics,
   decoder GroupNorm, contour-weighted BCE + Dice, auxiliary contour supervision,
   mixed precision, gradient accumulation, EMA and a completed cosine schedule.
6. **Native inference:** overlapping 1024 tiles blend with Hann weights. Four flip
   views and both encoders are averaged. Detector proposals are fused across flips.
   Proposal masks retain identity across fragments; dense components can recover
   detector misses. Exclusive ownership guarantees no output pixel overlaps.
7. **PQ selection:** fixed-rule native calibration PQ selects dense checkpoints every
   four epochs. A date-crossfitted quality regressor estimates expected credited IoU.
   Recipe selection compares semantic/hybrid, learned/simple filters, probability
   thresholds and minimum sizes. An adaptive family ranks alternative thresholds
   per instance using learned quality and suppresses nested duplicates. No report
   labels are fitted or tuned against.
8. **Full training data:** optional and enabled by default, a fresh refit uses the
   selected epoch counts on all competition training labels. Postprocessing and
   quality scoring stay frozen. The earlier report does not score these refitted
   models; refitting and the two-encoder average remain empirical choices.

The detector's checkpoint is selected by box mAP, then evaluated as part of final
PQ calibration. It is not claimed to be the best possible PQ detector checkpoint.
The quality target assumes independent proposal credit before final ownership;
overlap interactions can change actual credit. Calibration quality scores are
crossfitted, but dense checkpoint selection used the same calibration partition;
therefore calibration PQ is explicitly a selection score. The report partition is
the independent check. A single grouped split is less conclusive than repeated
grouped cross-validation; public scores should not be used for broad blind tuning.

## Running and artifacts

The notebook is standalone: it writes its embedded importable runtime before
launching training. Attach competition data, select **GPU T4 x2**, enable Internet
for packages/generic weights, and run the cells. Full defaults train the detector
and both mask models, calibrate, evaluate the report, refit on all training labels,
and generate `/kaggle/working/submission.csv`. Runtime on two T4s has not been
measured. Use the explicit smoke mode first for an environment check; it generates
only `smoke_submission.csv`, never a competition-ready file.

DDP uses two loader workers per rank (four total). Prediction uses one process per
GPU on disjoint image shards; CPU image/candidate tasks have four bounded workers.
The notebook logs actual rank devices, native crop shapes, worker counts, optimizer
updates and peak allocated VRAM. Smaller batches can address memory pressure without
changing image detail. The notebook does not automatically reduce image resolution.

Images are preprocessed once and hard-linked across annotator records. A preflight
measures compressed input sizes and budgets incremental cache, compact maps,
checkpoints and a reserve. It does not reserve 34 GiB or duplicate the full RGB
dataset. SHA256 fingerprints cover code, configuration, annotations, image content
and trained weights. Completed stages and maps are reusable only within their
matching run; older experiments are not recursively deleted.

The exporter verifies all 180 test images, native dimensions, unique original-stem
IDs, COCO compressed column-major RLE round trips and pixel non-overlap. Zero-mask
images are logged, not assigned fake masks. Outputs include a calibration grid,
frozen recipe, report PQ/SQ/RQ/TP/FP/FN, geometry-fallback cohort metrics, training
histories, previews, checkpoint provenance and a submission audit.

Validation is recorded separately in `Best/PQ50_VALIDATION.md`. Local smoke results
test execution mechanics, not segmentation quality or a >0.50 leaderboard claim.
