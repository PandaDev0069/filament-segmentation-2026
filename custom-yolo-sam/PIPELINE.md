# Pipeline guide

## Design and resolution contract

YOLO learns a single class, `filament`, from bounding boxes of merged training
masks. It supplies box prompts to SAM 2.1. SAM encodes the full image once and
decodes one or a few prompts at a time. A prompt box selects an object through
attention; it does not extract an image crop or truncate the returned mask.

The pipeline accepts only 2048 × 2048 source images. Dimensions are checked in
COCO metadata, decoded training/test images, YOLO batch preprocessing, model input
hooks, SAM encoding, loss targets and submission masks. Unexpected sizes raise
errors. No out-of-memory handler silently lowers the image size.

YOLO training uses `imgsz=2048` and byte-identical copies of the source images.
All mosaic, mixup, cutmix, copy-paste, affine scaling, translation, shear,
perspective and rotation settings are zero. Horizontal/vertical flips retain the
pixel grid. A small notebook-defined trainer also disables rectangular validation
padding; ordinary YOLO rectangular batches can exceed 2048 despite `imgsz=2048`.
Its validator checks actual batch dimensions. CUDA AMP remains enabled as requested. A scoped trainer override bypasses only
the SDK's unrelated lower-resolution compatibility probe; the standard scaler and
resume machinery remain active. The original probe is restored even on failure.

SAM is instantiated through Ultralytics but its high-level resizing predictor is
never called. `set_imgsz((2048, 2048))` configures prompt coordinates, embedding
dimensions and mask-input dimensions. RGB pixels receive SAM's normal mean/std
normalization and go directly to the image encoder. The 2048-pixel input produces
feature grids at 512, 256 and 128 pixels per side. Decoder logits are 512 × 512;
only logits are interpolated to native resolution. Images and ground-truth masks
are never interpolated.

Pretrained SAM 2's positional embeddings adapt to the larger feature grid.
This is a resolution adaptation of a pretrained model, not evidence of better
segmentation. Internal downsampling still limits detail. Inspect thin structures
on held-out predictions before spending the full training budget.

## Targets and split

The preparation cell validates IDs, source paths, dimensions, polygons/RLE and
nonempty annotations. It rejects unknown image references and crowd annotations.
COCO polygons, compressed RLE and uncompressed RLE are supported. All supplied
filament categories collapse to one foreground class.

For each filename, it ORs every annotation mask from every annotator. It labels
4- or 8-connected components and retains those meeting `min_component_area`.
Each retained mask is stored as compressed native COCO RLE; the same RLE determines
its detector box. The original annotations are retained separately in memory.
Preparation caches bind annotations, target settings and decoded source pixels.

Defaults (`connectivity=8`, `min_component_area=1`) retain the exact foreground
union, including small structures. A contact between filaments can join them.
Disconnected portions of one original annotation can become separate targets.
This ambiguity cannot be resolved by union alone. The preview shows original
annotator contours next to the resulting union. Changing connectivity is a
deliberate target-policy experiment; it is not an identity-matching algorithm.

Splitting groups decoded-pixel hashes, keeping identical images under different
filenames together. The default keeps all multi-annotator sources in training and
selects validation from the remaining groups. The validation fraction is a target
number of source groups, so duplicate groups or a limited eligible pool can alter
the realized fraction. The saved manifest records actual membership. Set
`multi_annotator_train_only=False` to permit held-out multi-annotator sources.

## Training

Two independent training jobs run concurrently in fresh interpreters: YOLO on the
first T4 and SAM on the second. There is no cross-model gradient dependency.
Weights are downloaded/resolved once before launching either job. This uses model
concurrency, not data-parallel training of one model across both GPUs. If one job
finishes first, that GPU waits until prediction; no speedup factor is assumed.

YOLO uses its standard detection objective and chooses `best.pt` by detector
validation fitness. The first configuration cell exposes optimizer, learning rate,
weight decay, epochs, image batch size, cache mode, patience and flips. Its settings
match the supplied recipe: AdamW, cosine decay, deterministic training, pretrained
weights, horizontal/vertical flips of 0.5, hue/saturation jitter zero, brightness
jitter 0.3, plots and verbose logs enabled. `max_det` also uses the central detection
limit. Rotation, translation, scaling, shear, perspective, mosaic, mixup and
copy-paste remain disabled. Native input dimensions remain fixed. Default
YOLO26 is an end-to-end detector; its usual inference does not use traditional
box NMS, so `predict.box_iou` mainly affects detector variants with a NMS head.

SAM trains the mask decoder while freezing the image encoder, prompt encoder,
video/memory modules, and `conv_s0/conv_s1` feature projections. Those projections
run with the frozen encoder pass. All other trainable parameters belong to the
mask decoder. This keeps native-resolution training practical without saving
backward activations through the full image encoder. The video memory is unused.

SAM uses a shared train/validation loader with two persistent CPU workers and a
replay sampler. Switching phases changes sampler indices instead of creating a
second pool. YOLO uses two training loader workers and zero extra validation
loader workers. Thus concurrent training requests four loader workers total.
Empty/tiny SDK datasets may use fewer workers.

Each image is encoded once. Every merged target receives its own padded box,
with optional pixel jitter during training. Targets and boxes share optional
horizontal/vertical flips. Native masks are decoded only for the current prompt
batch. The training loop visits all components; there is no hidden maximum target
count or subsampling. Empty-source images train the detector, but contribute no
SAM box-prompt loss.

The decoder objective is a weighted sum of:

- Pixel BCE with a configurable positive-class weight.
- Soft Dice loss on the native foreground probabilities.
- MSE between predicted quality and detached measured mask IoU.

The quality prediction is supervised, but final filtering uses detector confidence,
mask probability and mask NMS. SAM quality is not assumed to be calibrated for
solar images. Loss weights are configurable. Loss is averaged across components
per image and then across image accumulation groups, including the final partial
group. AMP scaling and gradient clipping are applied before optimizer updates.

Validation loss uses held-out merged masks and their ground-truth boxes. The best
decoder minimizes this loss. This isolates segmentation learning from detection,
but is not the final combined-system PQ. That score is computed afterward using
actual YOLO boxes. Set the final model's learning rate/epochs based on this
distinction; better prompt loss cannot compensate for missed detector proposals.

## Validation and threshold selection

For validation and test inference, each T4 loads a complete YOLO + SAM replica.
Uncached source images are sharded between GPUs without duplication; byte-identical
files sharing a cache key are scheduled once. Within each GPU process, a bounded
two-thread pool overlaps one native image read with compressed-cache writing while
the coordinator performs model forwards. Pending cache writes are bounded and
must finish before the stage succeeds. GPU inference stays in the coordinator,
which avoids calling one model instance concurrently from multiple threads.

For each held-out source, prediction uses a low configurable proposal confidence.
Native SAM logits are cached as float16 at decoder resolution, together with boxes
and detector scores. Cache keys include source file content, checkpoint hashes,
SAM base weights and inference settings. The search uses only validation data. CPU search uses four threads over sources,
loading one source's logits per worker rather than retaining all validation logits
in memory. Merged-mask preparation, source indexing, detector image export,
selected-metric scoring and submission RLE encoding use the same bounded pool
in separate phases; four extra pools do not run simultaneously.

The Cartesian search varies detector confidence, mask threshold, mask NMS IoU and
minimum mask area. Do not search a detector confidence below `proposal_conf`:
proposals below that floor were never decoded. Thresholding occurs after native
logit interpolation. Mask NMS suppresses duplicate instances by mask IoU; distinct
overlapping predictions remain independent. Disconnected pieces of a predicted
instance stay together. There is no default hole filling, erosion, dilation,
largest-component selection, or clipping to the prompt box.

PQ uses each original annotator-image record as the reference, even though
training uses merged masks. It computes independent-mask COCO IoUs, counts **all**
pairs with IoU **strictly greater than 0.5**, and counts unmatched prediction
columns/GT rows as FP/FN. The counts and matching IoUs sum globally:

```text
PQ = sum(matched IoUs) / (TP + 0.5 * FP + 0.5 * FN)
```

This follows the supplied `self-evaluation-notebook.ipynb`, including the unusual
possibility of matching one prediction to multiple overlapping GT instances.
No Hungarian assignment or mean of per-image PQ is substituted. Empty cases are
handled explicitly; an all-empty aggregate has score zero. Merged foreground
Dice/IoU are additional diagnostics, not replacements for original-instance PQ.
If area filtering is enabled, these foreground diagnostics use the retained merged
training targets, as the column names and review plots indicate.

`threshold_trials.csv` records all trials. Stable ordering resolves tied scores.
The selected JSON binds thresholds to checkpoint hashes. Repeated tuning on the
same validation split can overfit it; use a separate held-out evaluation when
comparing many experiments.

## Submission and recovery

Immediately after training, the notebook displays YOLO's returned validation
metrics, last-epoch losses/metrics, best recorded box mAP50?95, SAM loss tables and
its loss plot. A grouped gallery follows the supplied plot categories and layouts,
skipping absent files from short runs. YOLO26 is anchor-free; the training-batch
group displays batch images. YOLO segmentation `Mask*` curves are not generated by
this box detector. SAM quality appears later as combined validation PQ, Dice and
IoU, with its prediction overlays. Plot generation is synchronous inside the GPU
coordinator, avoiding additional background plotting threads.

The final CSV has exactly `filament_id` and `segmentation_rle`. IDs use the original
test filename stem followed by a 1-based instance number. RLE is compressed COCO
counts in Fortran order. Masks are native 2048 × 2048, and every row is decoded
and re-encoded before acceptance. Original filenames are never inferred from
annotator IDs. Test stems must be unique.

Four CPU threads build and validate per-source journals in parallel. One writer
consumes their results in deterministic source order and atomically replaces the
CSV. Completed journals bypass raw prediction generation during resume.

Zero predictions produce no CSV rows. The manifest records an explicit count for
every processed source, including zero. Header-only output is allowed when all
predictions are empty; the notebook prints the zero-source count. It never
invents fallback masks. Journals bind each completed source to image content,
checkpoint provenance and selected thresholds. Inference can resume without
repeating completed sources; all sources must finish before atomic CSV replacement.

Use `stages.train_yolo=False` and/or `stages.train_sam=False` to reuse this run's
checkpoints. Ordinary reruns also reuse existing best checkpoints. To recover an
interrupted training run, enable the respective `resume` setting. SAM saves its
decoder, optimizer, scaler, epoch, best loss and patience state. YOLO uses its
native training checkpoint recovery. A completed YOLO run cannot be extended
with `resume=True`; start a new run for a new schedule.

Change `run_name` for altered training hyperparameters or targets. A training
contract rejects mismatched settings; SAM additionally checks its base-weight
hash. Epoch count can change for interrupted recovery, but an existing completed
best checkpoint is reused unless resume is requested. Do not edit configuration
mid-run: restart and run from the beginning so fingerprints are recomputed.
Smoke/full artifacts are isolated and SAM refuses mismatched checkpoint modes.

## Practical settings

| Setting | Effect |
|---|---|
| `yolo.weights`, `sam.weights` | Smaller models reduce compute and memory; local paths support offline runs |
| `yolo.batch` | Full-image detector batch on its assigned T4; default 2 |
| `resources.cpu_workers` | Total CPU job/loader budget; default 4, split 2 + 2 for concurrent GPU jobs |
| `sam.prompt_batch` | Simultaneous object decodes and native-mask loss tensors; default 2 |
| `sam.accumulate_images` | Effective image accumulation without larger encoder batches |
| `sam.box_pad`, `box_jitter` | Prompt tolerance in original pixels; no image cropping |
| `predict.proposal_conf` | Recall/compute tradeoff before SAM; missed proposals cannot be recovered |
| `predict.max_det` | Cap on predicted boxes per source; increase if validation reveals truncation |
| `search` | Validation-only parameter grid; larger grids increase CPU evaluation time |
| `merge.min_component_area` | Training target filtering; default 1 preserves small components |

The default profile targets Kaggle GPU T4 x2, each with 15 GiB VRAM. A configurable
13 GiB PyTorch allocator cap applies separately to each GPU, with no pooled memory.
FP16 is enabled; BF16 and TF32 are not required. The GPU workers start as fresh
interpreters with one visible device each. Notebook definitions are exported as
JSON runtime bundles; separate Python modules and custom-trainer pickles are not
needed. Linux CPU loader workers use `fork` and never create CUDA objects.

The notebook process keeps metrics/preparation on CPU and releases GPU worker
processes between stages. Per-job logs and periodic status make failures visible;
failed stages leave usable caches/checkpoints for recovery. The CPU and loader
thread counts are bounded, although Python coordinators and CUDA housekeeping
threads also exist and are not counted as loader workers.

The dual-T4 profile, its memory ceiling and its throughput have not been tested
here. Reduce image/prompt batch sizes if necessary, retaining 2048-pixel input.
Using `gpu_ids=[0]` runs training jobs sequentially. The current profile requires
Linux. Frozen-encoder training cannot learn new image features; that remains an
explicit memory/complexity tradeoff.

## References

- [Ultralytics training API](https://docs.ultralytics.com/modes/train/)
- [Pinned SAM model implementation and size configuration](https://github.com/ultralytics/ultralytics/blob/v8.4.198/ultralytics/models/sam/modules/sam.py)
- [Pinned SAM predictor feature/prompt implementation](https://github.com/ultralytics/ultralytics/blob/v8.4.198/ultralytics/models/sam/predict.py)
- [Meta SAM 2](https://github.com/facebookresearch/sam2)
- Scoring reference: the repository's supplied `self-evaluation-notebook.ipynb`

The dependency's source and model licenses apply to their use. The notebook does
not bundle pretrained weights or the competition dataset.
