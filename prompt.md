# Codex Task: Build a YOLO26 Crop-Based Training Pipeline for Solar Filament Segmentation Challenge 2026

## 1. Objective

Create a new, complete, executable Jupyter notebook named:

`yolo26-crop-training-pipeline.ipynb`

The notebook must implement an end-to-end instance segmentation pipeline using Ultralytics YOLO26 segmentation models for the Solar Filament Segmentation Challenge 2026.

The main experiment is to improve solar filament segmentation by creating fixed-size training crops from individual filament annotations, treating different annotators' annotations of the same physical image as separate annotation views.

The pipeline must include:

1. Dataset discovery and annotation auditing.
2. Leakage-safe training and validation splitting.
3. Annotation-specific crop generation.
4. Filament-centered positive samples.
5. Regular grid/context samples.
6. Empty-background negative samples.
7. Conversion of instance masks into YOLO segmentation labels.
8. Pretrained YOLO26 segmentation model training.
9. Full-resolution validation using tiled inference.
10. Reconstruction of predictions into original-image coordinates.
11. Instance stitching and configurable post-processing.
12. Official competition evaluation, visualization, and metric reporting.
13. Test inference and valid submission CSV generation.
14. Reproducibility, caching, checkpoint saving, and final submission auditing.

Do not modify or overwrite `yolo-post-processing.ipynb`. Use it as a reference for the existing dataset structure, annotation handling, official evaluation implementation, output format, and submission requirements.

Produce the actual `.ipynb` file, not merely a code outline or a textual explanation.

## 2. Inspect the Existing Project First

Before writing the notebook, inspect the available files and relevant notebooks in the current workspace.

Use these resources where available:

- `yolo-post-processing.ipynb`: existing prediction processing, annotator-aware validation records, evaluation functions, and submission generation.
- The competition's official Self Evaluation Notebook: authoritative evaluation behavior.
- The existing YOLO26 training notebook or configuration, if available: model naming, augmentation settings, and training conventions.

Reuse proven components where appropriate instead of reimplementing them unnecessarily.

Do not blindly copy an incorrect implementation. Verify the annotation mapping, metric computation, and instance identifiers against the official evaluation implementation.

Competition reference:

https://www.kaggle.com/competitions/filament-segmentation-2026/overview

YOLO26 documentation:

https://docs.ultralytics.com/models/yolo26/

If a referenced file is not available, identify the missing file and use an available, verified alternative. Do not pretend that a reference was inspected when it was not.

## 3. Dataset Discovery and Structure

The expected dataset resembles:

`MAGFiLO_1.0_Kaggle_2026/`
- `train/`
  - `train_images/`
  - `MAGFiLO_1.0_Annotations_kaggle2026_train.json`
- `test/`
  - `test_images/`

Typical original images have dimensions of 2048 × 2048 pixels.

Automatically discover the dataset root where possible. Do not hardcode a single Kaggle input path if the notebook can robustly discover the actual dataset location.

Use the COCO annotation structure and inspect it before making assumptions about annotation ownership.

Each annotation must be associated with the correct annotation view, using the COCO `image_id` and any verified annotator-specific metadata.

### Important distinction: physical image versus annotation view

Several COCO image records may reference the same physical JPEG, while representing independent annotations from different annotators.

For example:

| Physical file | Annotation view | Label source |
|---|---|---|
| `image_001.jpeg` | View A | Only annotations associated with A |
| `image_001.jpeg` | View B | Only annotations associated with B |
| `image_002.jpeg` | View C | Only annotations associated with C |

The physical image is shared, but the ground-truth annotation sets must remain separate.

Requirements:

- Inspect how annotator-specific information is represented in the actual JSON.
- Preserve the correct image-record and annotation relationships.
- Never combine conflicting annotations from different annotators into a single training label set.
- Do not count multiple annotation views as independent physical images when splitting the data or reporting physical-image counts.
- Report the number of physical images, annotation views, and individual filament instances separately.
- Validate that each annotation belongs to the intended image record.

## 4. Leakage-Safe Training and Validation Split

Split the original annotated dataset before generating any crops.

Use a reproducible 80/20 training-validation split, seed 42.

Prefer reproducing the same filename-grouped `GroupShuffleSplit` procedure used in `yolo-post-processing.ipynb`, including its input record ordering, so the new model can be compared fairly against the existing baseline.

Grouping key:

`image["file_name"]`

All annotation views of the same physical image must belong to the same split.

All crops generated from those views must inherit that split.

Enforce:

`train_physical_filenames ∩ validation_physical_filenames = ∅`

Save a split manifest containing the physical filename, COCO image ID, annotation-view identifier, and assigned split.

Log the number of physical images and annotation views in each split.

Do not split individual annotation records or generated crops independently.

## 5. Crop Generation Pipeline

Create a configurable preprocessing pipeline that converts the original 2048 × 2048 images into smaller, fixed-aspect-ratio training samples.

Initial configuration:

- `CROP_SIZE = 640`
- Output crop dimensions: 640 × 640
- Aspect ratio: 1:1
- `SEED = 42`
- `VALIDATION_FRACTION = 0.20`
- Configurable crop overlap and edge handling
- Configurable minimum visible mask area

Use the original-resolution image as the source. Do not distort or stretch filament structures to force them into a square.

When a crop reaches the image boundary, pad the missing region appropriately.

### 5.1 Filament-centered crops

Generate positive crops centered around each individual annotated filament.

For each annotation view:

1. Read the source image.
2. Retrieve only the annotations associated with that annotation view.
3. Decode each filament's mask.
4. Calculate its bounding box and center.
5. Generate one or more fixed-size crops that capture the filament and surrounding context.
6. Save the crop and corresponding YOLO segmentation labels.
7. Record the source filename, annotation-view identifier, source annotation ID, crop coordinates, and generated crop ID in a manifest.

Do not simply resize an entire 2048 × 2048 image to 640 × 640 for these samples.

### 5.2 Long and large filaments

Some filament masks may extend beyond the size of one crop.

Do not shrink these filaments until their fine structures become too small to learn.

Instead, generate overlapping crops that cover the complete annotated structure.

Ensure that long filaments and filaments close to image boundaries are not systematically excluded.

Avoid generating an excessive number of nearly identical crops for one target. Use configurable limits, deduplication, and coverage checks.

### 5.3 Preserve all visible labels in each crop

A crop must contain the correct labels for every annotated filament visible within its boundaries in the same annotation view.

For example, if a crop is centered on filament A but also contains part of filament B, include both A and B if their visible masks meet the configured minimum visibility requirements.

Never label a region containing a known, labeled filament as background merely because that filament was not the target used to generate the crop.

Clip masks to the crop boundaries and transform them into crop-local coordinates.

Every label must refer to the same annotation view used to create that crop.

### 5.4 Mask-to-polygon conversion

Convert the cropped binary instance masks to the polygon representation required by YOLO instance segmentation.

Requirements:

- Use correct crop-local coordinates.
- Normalize coordinates to the crop dimensions.
- Ensure normalized values remain within `[0, 1]`.
- Reject invalid or degenerate polygons.
- Preserve thin structures as much as possible.
- Use configurable polygon simplification rather than aggressive contour simplification.
- Account for masks containing multiple disconnected components.
- Do not silently discard mask components or merge separate filaments.
- Report cases where polygon conversion cannot faithfully represent a mask.

Save representative original masks and reconstructed polygon masks for visual verification.

Calculate and report the mask reconstruction IoU for a sample of generated labels.

Clearly document any limitations of converting raster masks to polygon-based YOLO labels.

### 5.5 Regular grid/context crops

Also generate regular grid crops from training images.

These teach the model to recognize filament structures outside crops that are deliberately centered on a known target.

Use an overlapping tile grid with edge coverage.

Initially, use 640 × 640 crops with a configurable stride, such as 480 pixels.

Label each grid crop using all qualifying annotations from its own annotation view.

Include partially visible filaments when their masks remain sufficiently informative.

### 5.6 Empty-background crops

The model must learn when there is no filament to predict.

Generate negative crops from regions that contain no annotated filaments.

Possible sources:

- Random background locations.
- Empty grid crops.
- Challenging background regions near non-filament solar structures.

A true negative crop must receive an empty YOLO label file.

Do not introduce a separate background class. This is a single-class instance-segmentation problem.

To reduce contradictory supervision from annotators, implement a configurable safety mechanism that excludes candidate negative crops overlapping any annotated filament region from any annotation view of the same physical image.

Use the other annotators' masks only to identify potentially unsafe negative regions. Do not merge their masks into the current annotation view's training labels.

For example:

`EXCLUDE_ANY_ANNOTATED_FILAMENT_FROM_NEGATIVES = True`

Apply a configurable safety margin around annotated masks.

If the candidate crop overlaps a safety region, reject it as a negative sample or regenerate another crop.

Audit the negative crops visually and verify that they do not contain known filament structures.

### 5.7 Initial sampling balance

Make the sampling strategy configurable.

Start with approximately:

- 50% filament-centered crops.
- 25% grid/context crops.
- 25% empty-background crops.

Treat these as initial sampling targets rather than mandatory exact percentages.

The physical crop dataset may be imbalanced because some images contain many more filaments than others.

Report:

- Total generated crops.
- Positive crops.
- Negative crops.
- Filament instances.
- Crops per annotation view.
- Crops per physical source image.
- Distribution of visible mask areas.
- Distribution of crop-generation strategies.

Avoid allowing a small number of heavily annotated images to dominate the training dataset.

Make the pipeline deterministic and prevent duplicate crop generation when the preprocessing cells are rerun.

## 6. Disk-Based Dataset and Caching

Generate a YOLO-compatible dataset under a configurable output directory, preferably `/kaggle/working/` when running on Kaggle.

Use a structure similar to:

`yolo_filament_crop_dataset/`
- `images/`
  - `train/`
  - `val/`
- `labels/`
  - `train/`
  - `val/`
- `dataset.yaml`
- `crop_manifest.csv`
- `split_manifest.csv`
- `dataset_audit.json`

Each crop must have a corresponding label file.

For negative crops, create an empty `.txt` file.

Use unique crop identifiers so multiple annotation views of the same JPEG cannot overwrite one another.

The manifest should preserve the relationship between:

- Physical filename.
- COCO image ID.
- Annotation view.
- Source annotation IDs.
- Original crop coordinates.
- Crop strategy.
- Split.
- Output image and label paths.

Use disk caching and incremental preprocessing where appropriate.

If a matching dataset manifest already exists, verify that its source files and preprocessing configuration are unchanged before reusing it. Otherwise, regenerate the affected outputs.

Do not keep the entire decoded dataset in RAM unnecessarily.

## 7. Dataset Visualization and Preflight Checks

Before training, display examples of:

- Original images with all annotations from one view.
- The same image under two different annotation views.
- Individual filament-centered crops.
- Overlapping crops of long filaments.
- Grid/context crops.
- Negative crops.
- Crops containing multiple instances.
- Crops near image edges.
- Crops with partially visible instances.

Display mask overlays and polygon labels.

Implement automated checks for:

- Missing or unreadable images.
- Missing annotations.
- Invalid image dimensions.
- Incorrect image-ID-to-annotation mapping.
- Duplicate crop IDs.
- Broken label paths.
- Empty or malformed label files.
- Coordinates outside the crop.
- Degenerate polygons.
- Positive crops with missing expected masks.
- Negative crops that violate the background safety rule.
- Train/validation filename leakage.
- Annotation views assigned to different splits for the same physical image.
- Unexpected crop-size or aspect-ratio mismatches.

Fail clearly on serious data-integrity errors instead of continuing silently.

## 8. YOLO26 Instance Segmentation Training

Use a pretrained Ultralytics YOLO26 segmentation checkpoint:

`yolo26s-seg.pt`

Do not silently substitute YOLOv8, YOLO11, a detection-only model, or a semantic-segmentation-only model.

Check that the installed Ultralytics version supports the required model and task. If the checkpoint is missing or incompatible, report the problem clearly.

Define one class:

`filament`

Create a valid YOLO dataset YAML referencing the generated train and validation images.

Initial training configuration:

- Model: YOLO26s segmentation.
- Input image size: 640.
- Epochs: configurable, starting at 100.
- Batch size: configurable, starting at 8.
- Workers: configurable, starting at 4.
- Device: automatically detect CUDA; default to GPU 0 when available.
- Early stopping patience: 20.
- Random seed: 42.
- Deterministic settings: configurable and logged.
- Disk caching: enabled when supported and appropriate for the runtime.
- Checkpoint saving: enabled.
- Plot generation: enabled.

The batch size, number of workers, and cache mode must be configurable for different Kaggle accelerator and RAM limits.

Use conservative, scientifically reasonable augmentation.

Prefer transformations that preserve filament structure. Avoid aggressive crops, geometric distortions, mosaic, mixup, or copy-paste unless they are explicitly introduced as separate experiments and verified not to corrupt the masks.

Because the input images are grayscale H-alpha observations, preserve their grayscale appearance. If the installed YOLO pipeline expects three channels, convert grayscale images to three identical channels without introducing artificial color.

Log:

- Python version.
- PyTorch and CUDA versions.
- Ultralytics version.
- Available GPU names and memory.
- Dataset configuration.
- Training hyperparameters.
- Checkpoint path.
- Epoch and training duration.
- Training and validation loss curves.
- Standard YOLO segmentation validation metrics available from the library.

Save the best checkpoint and last checkpoint.

Support both a short smoke-test configuration and a full training configuration. The smoke test should verify dataset loading and training for a minimal number of epochs without replacing the final training configuration.

Do not claim that training succeeded unless it was actually executed.

## 9. Full-Image Validation Using Tiled Inference

Standard validation on small training crops is insufficient to determine whether the model performs well on original solar observations.

The primary validation must operate on complete, original-resolution images.

For each unique physical validation image:

1. Load the original 2048 × 2048 image.
2. Generate overlapping 640 × 640 inference tiles.
3. Ensure complete coverage, including all image boundaries.
4. Run YOLO26 segmentation on each tile.
5. Map the predicted masks back into the original-image coordinate system.
6. Stitch duplicate detections across overlapping tiles.
7. Perform configured post-processing.
8. Reconstruct the complete set of predicted instances for the original image.
9. Evaluate the predictions against the appropriate ground-truth annotation views.

Use a configurable tile stride, initially around 480 pixels for 640-pixel tiles.

Do not use known filament bounding boxes or annotation-derived crop coordinates for validation inference. Validation must simulate the actual inference situation, where the filament locations are unknown.

Perform inference once per unique physical validation image, even if several annotators supplied annotations for it.

### Stitching requirements

Implement a separate, testable stitching function.

Requirements:

- Convert tile-local masks to original-image coordinates.
- Remove duplicate detections in tile-overlap regions.
- Avoid merging nearby but distinct filament instances.
- Use configurable mask overlap, IoU, confidence, and other justified matching criteria.
- Make the order of processing and tie-breaking deterministic.
- Ensure final instance masks have the original image dimensions.
- Avoid automatically bridging gaps unless a separate, explicitly enabled experiment demonstrates a benefit.
- Preserve confidence and diagnostic information where useful.

The stitching algorithm must not assume that all nearby filaments belong to the same instance.

Create visual examples showing tile boundaries, individual tile predictions, stitched predictions, and the final full-image masks.

## 10. Annotator-Aware Official Validation

Use the same official PQ evaluation behavior as the current competition Self Evaluation Notebook.

Start with the evaluation functions and annotation-handling logic in `yolo-post-processing.ipynb`, verifying their behavior against the current official implementation where available.

Do not replace the competition evaluator with generic Dice, mAP, or an independently implemented PQ formula as the primary score.

Important rules:

- Keep each ground-truth annotation view separate.
- Never union different annotators' masks for scoring.
- Do not score annotation views as if they were different physical images when reporting physical-image counts.
- Infer each unique physical validation image once.
- Compare the same reconstructed prediction set independently against each corresponding ground-truth annotation view, following the official evaluator's expected identifiers and matching semantics.
- Preserve the evaluator's treatment of overlaps, fragmentation, over-merging, false positives, and false negatives.
- Use exactly the image and mask dimensions expected by the official evaluator.

Be particularly careful about the distinction between COCO `image_id`, physical `file_name`, and submission `filament_id`.

Do not assume that annotation-view IDs can be substituted directly into submission IDs.

### Metrics to report

At minimum, report:

- Official Panoptic Quality (PQ).
- Segmentation Quality (SQ), if supported by the official evaluator.
- Recognition Quality (RQ), if supported by the official evaluator.
- True positives.
- False positives.
- False negatives.
- Precision and recall, where mathematically defined.
- IoU statistics.
- One-to-many overlap diagnostics.
- Many-to-one overlap diagnostics.
- Runtime for preprocessing, training, validation inference, stitching, and evaluation.

Save a CSV containing per-annotation-view metrics and a separate summary grouped by physical image.

Also show representative visual comparisons containing:

- Original H-alpha image.
- Individual ground-truth annotation views.
- Final predicted instances.
- False-positive examples.
- Missed filaments.
- Fragmented filaments.
- Incorrectly merged filaments.

Treat full-image official PQ as the primary comparison metric. Standard YOLO crop-level metrics are supplementary.

## 11. Compare Against the Existing Baseline

The existing pipeline reported a validation PQ of approximately 0.4213 for its selected post-processing configuration.

Use the same held-out physical images and the same official evaluator when comparing the new crop-based model against that baseline.

If the baseline checkpoint is available, optionally run it through the same tiled-inference evaluation path. Make its path configurable and do not make the new notebook dependent on the baseline checkpoint being present.

Report both results in a comparison table containing:

- Model and checkpoint.
- Training strategy.
- Validation split identifier.
- Official PQ.
- SQ and RQ where available.
- TP, FP, and FN.
- Inference and reconstruction runtime.

Do not claim that the crop-based method is better based only on increased training-data size.

If the baseline and crop-based pipeline use different validation splits or different evaluation behavior, identify that limitation explicitly rather than presenting the scores as a fair comparison.

## 12. Configurable Post-Processing Experiments

Keep post-processing separate from the model so that strategies can be tested without retraining or repeating raw inference.

Cache the raw tiled predictions, including masks, confidence values, class IDs, and tile coordinates.

Provide configurable post-processing parameters, including:

- Confidence threshold.
- Minimum instance area.
- Duplicate-removal thresholds.
- Cross-tile stitching thresholds.
- Optional connected-component handling.
- Optional morphology, when justified.

Start with a conservative baseline configuration.

Then evaluate a small, configurable collection of alternatives against the same cached predictions and official evaluator.

Save the experiment results and selected configuration.

Choose the best configuration using validation data only.

Once selected, freeze the configuration. Do not tune it using test predictions or test ground truth.

## 13. Final Test Inference

After training and validation, run inference on every original image in `test/test_images`.

The test path must use only the permitted test H-alpha images for inference.

Do not generate test crops using ground-truth masks, annotation-derived bounding boxes, filenames that encode target locations, external ground-truth metadata, or any other unavailable test labels.

Use the same tiled-inference, stitching, mask-reconstruction, and frozen post-processing procedure validated on the held-out images.

The test set must remain a set of original physical images. Multiple tiles are inference inputs, not additional physical test images.

Cache test predictions where practical so the notebook can resume after interruptions.

Never load or manufacture test labels.

## 14. Submission Generation

Produce:

`/kaggle/working/submission.csv`

Use exactly these columns:

- `filament_id`
- `segmentation_rle`

Generate one row for each predicted filament instance.

Use the original physical image's official filename stem as the image identifier and assign unique instance suffixes according to the competition's expected format.

Encode every binary predicted mask using `pycocotools.mask.encode` on a Fortran-contiguous array, and convert the compressed RLE counts into the string format expected by the competition.

Verify that reconstructed masks use the original 2048 × 2048 coordinate system.

Implement a final audit that checks:

- Exact CSV column names and order.
- No missing or empty IDs or RLE values.
- No duplicate `filament_id` values.
- Every ID references an actual test image.
- No train or validation image is accidentally included.
- Every RLE decodes successfully.
- Every decoded mask has the expected dimensions.
- Masks are nonempty and valid.
- The number of instances is reported per physical test image.

Run the audit on all rows where practical, not just the first row.

If no detections are produced for an image, do not invent a filament mask. Handle the absence of instances according to the competition's accepted submission format.

## 15. Reproducibility and Runtime Reliability

Organize the notebook into clear, numbered sections:

1. Introduction and pipeline overview.
2. Configuration.
3. Imports and environment checks.
4. Dataset discovery.
5. COCO annotation and annotator-view audit.
6. Filename-grouped train/validation split.
7. Crop-generation utilities.
8. Training dataset generation.
9. Dataset visualization and preflight checks.
10. YOLO26 dataset configuration.
11. Model training.
12. Training-curve and metric analysis.
13. Full-image tiled validation inference.
14. Cross-tile stitching.
15. Official PQ evaluation.
16. Post-processing experiments.
17. Baseline comparison.
18. Frozen best configuration.
19. Test inference.
20. Submission generation.
21. Final submission audit.
22. Reproducibility notes and known limitations.

Keep functions modular and reusable.

Avoid unnecessarily large monolithic cells, excessive inline comments, and unnecessary dependencies.

Use progress bars for long-running operations.

Use exception handling to provide helpful messages when files, checkpoints, dependencies, or expected columns are missing.

Make preprocessing, prediction caching, and training outputs restartable where practical.

Avoid holding multiple copies of the entire dataset in RAM. Use disk-backed caches and bounded batches where appropriate.

Save important manifests, configurations, metrics, and checkpoint paths to `/kaggle/working/`.

Do not delete the original images, annotations, existing notebooks, or source checkpoints.

## 16. Acceptance Criteria

The notebook is complete only when:

1. The actual `.ipynb` file has been created.
2. The notebook contains the full preprocessing, training, validation, and submission pipeline.
3. Multiple annotator views of one physical image are treated independently for labels.
4. Training and validation are split by physical source image before crop generation.
5. All crops use the expected fixed dimensions and valid labels.
6. Negative crops have empty labels and pass the configured background checks.
7. Long filaments and image boundaries are handled.
8. A YOLO26 segmentation model can be trained using the generated dataset.
9. Validation inference reconstructs full-size predictions without accessing validation annotation locations.
10. Evaluation uses the official competition methodology and preserves separate annotator-specific ground truth.
11. The notebook reports official PQ and useful diagnostic metrics.
12. Test inference uses only permitted test inputs.
13. The final submission has the correct schema, IDs, and RLE encoding.
14. Syntax and basic data-integrity tests pass.
15. Any untested steps, limitations, or remaining execution requirements are stated honestly.

Before finishing, inspect the complete notebook for undefined variables, missing imports, incorrect paths, inconsistent function signatures, invalid YOLO labels, and cell-order dependencies.

Run available syntax checks and lightweight preprocessing tests. Do not start an expensive full training run merely to claim completion if the environment does not support it.

Finally, summarize the notebook created, its main design decisions, the tests actually run, and any remaining execution steps. Do not silently invent results or claim unexecuted tests passed.



## Critical Architecture Requirement: Separate Training from Validation and Inference

We have limited GPU execution time. Therefore, the new training notebook and the existing post-processing notebook must have strictly separate responsibilities.

### A. New notebook: `yolo26-crop-training-pipeline.ipynb`

This notebook is responsible for **preprocessing, dataset generation, and YOLO26 training only**.

It must:

1. Audit the COCO dataset and identify independent annotation views using `image_id`.
2. Split physical images into training and validation groups before generating crops. All annotations and crops associated with the same physical filename must remain in the same split.
3. Generate annotation-specific filament-centered crops, contextual grid crops, and verified empty-background crops.
4. Convert the cropped instance masks into valid YOLO segmentation labels.
5. Train a pretrained `yolo26s-seg.pt` model using only crops derived from the training split.
6. Save the crop manifest, split manifest, dataset configuration, training configuration, training logs, and model checkpoints.

**Do not perform model validation or inference in this notebook.**

Specifically:

- Set YOLO training's `val=False` where supported by the installed Ultralytics version.
- Do not call `model.val()`.
- Do not run tiled inference.
- Do not reconstruct full-image predictions.
- Do not calculate competition PQ or select post-processing thresholds.
- Do not generate a test submission.
- Do not use validation metrics for early stopping because validation is disabled.

Use a configurable fixed epoch count instead of validation-dependent early stopping. Save `last.pt` as a mandatory checkpoint and save other checkpoints when available. Do not assume that `best.pt` necessarily represents the best validation model when validation is disabled.

CPU-based preprocessing and training-loss reporting are allowed. GPU usage should be limited to the actual training process.

### B. Artifacts handed to the post-processing notebook

Save all required artifacts to a configurable persistent output directory.

At minimum, export:

- The trained checkpoint, such as `yolo26s_crop_training.pt`.
- The exact model architecture and training configuration.
- The dataset and crop configuration.
- The physical-filename-based split manifest.
- The original COCO annotation mapping and annotation-view identifiers.
- Training logs and loss curves.
- The Ultralytics version and relevant environment details.

Ensure that these artifacts can be downloaded or accessed by the separate post-processing notebook. Do not rely on temporary in-memory variables shared between notebooks.

### C. Existing notebook: `yolo-post-processing.ipynb`

This notebook remains responsible for **all model validation, post-processing experiments, and test inference**, using the CPU only.

Update it to load the trained checkpoint from the exported artifact path and explicitly use CPU inference.

It must:

1. Load the saved physical-image train/validation split manifest.
2. Perform tiled inference on original-resolution validation images without using ground-truth locations.
3. Reconstruct full-image instance masks and stitch overlapping tile predictions.
4. Evaluate the reconstructed predictions against each annotator's independent ground-truth view using the verified official competition evaluator.
5. Run post-processing experiments on cached raw predictions.
6. Select and freeze the best configuration using validation PQ.
7. Perform CPU inference on the original test images.
8. Generate and audit the final submission CSV.

Cache predictions and make processing restartable to avoid repeating expensive CPU inference.

Do not train or fine-tune the model in the post-processing notebook.

The final workflow must be:

**Training notebook (GPU): annotation-aware cropping → YOLO26 training → export artifacts.**

**Post-processing notebook (CPU): load checkpoint → validation → post-processing selection → test inference → submission.**

Preserve the separation between the notebooks. Do not silently reintroduce validation or inference into the training notebook.