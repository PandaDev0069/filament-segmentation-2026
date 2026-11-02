# Validation evidence

## Current Kaggle revision: CPU checks only

The notebook now targets **two T4 GPUs with 15 GiB VRAM each and four CPU workers
in total**. This hardware configuration has **not been executed or benchmarked
here**. No CUDA model execution or new training run was performed for this update.
Linux multiworker loader execution and the 13 GiB per-GPU allocator budget remain
unverified on Kaggle.

On **2026-11-02**, 13 CPU check groups passed against the delivered notebook.
[validation/kaggle_cpu_checks.json](validation/kaggle_cpu_checks.json) records its
current SHA-256, scope and results:

- Notebook schema, Python syntax, clean outputs and resource defaults.
- Existing annotation-format, union-mask, native-size, PQ and RLE contracts.
- Four-worker allocation and bounded, ordered CPU threading with propagated errors.
- GPU visibility mapping and numerical-library thread caps, checked as configuration.
- Exported notebook definitions imported in a fresh CPU-only interpreter.
- Native 2048-pixel CPU loading and reuse of sampler indices across phases.
- Requested YOLO training arguments checked with a mock model, including AdamW,
  cosine decay, deterministic mode, HSV brightness 0.3, flips, plots and verbosity.
- YOLO's worker allocation, zero extra validation workers, and restoration of the
  scoped AMP-probe override after an intentional setup error.
- Inference sharding, completed-cache reuse and deduplication of byte-identical files.
- Real native-image prefetch and atomic compressed-cache writing using mocked model
  forwards; these were CPU pipeline checks, not GPU inference tests.
- Metric tables and grouped galleries with partial/absent plot files.
- Actual fresh CPU subprocess overlap, worker budgets, error propagation and peer
  termination using intentionally failing mock jobs.

These checks verify orchestration and contracts. They do not establish dual-T4
memory fit, model accuracy, throughput, or Linux worker/CUDA behavior. Start with
`MODE='smoke'` after selecting Kaggle **GPU T4 x2** and restarting the kernel.

## Historical single-GPU revision

The evidence below belongs to the previous notebook revision. Its recorded
notebook hash in `validation/checks.json` intentionally differs from the current
Kaggle revision; it is retained as historical model/pipeline evidence.

Verified locally on **2026-11-02**, using the real MAGFiLO dataset and an NVIDIA
GeForce RTX 3070 Ti Laptop GPU with 8 GiB VRAM. These are execution checks, not
competition performance estimates. No full training run was performed.

Historical machine-readable evidence, including the previous notebook hash and checkpoint
provenance, is in [validation/checks.json](validation/checks.json).

## Completed checks

| Check | Result |
|---|---|
| Notebook schema and Python syntax | Passed; 20 cells, 11 executable code cells |
| Clean delivered notebook | Passed; no stored outputs or executed state |
| Full smoke workflow | All 11 code cells executed without errors |
| Real source audit | 707 sources, 1,154 annotator records, 8,199 annotations; 296 multi-annotator sources |
| Leakage grouping | All sources hashed; no identical-pixel groups found in this dataset |
| Target preparation | Two training and two validation sources; 20 merged components total |
| YOLO training and validation | One epoch at native 2048 × 2048; best/last checkpoints saved |
| SAM native probe | Real pretrained forward: input `[1,3,2048,2048]`, decoder logits `[1,512,512]` |
| SAM decoder training | One epoch with native-mask loss and frozen encoder; best/last checkpoints saved |
| Configurable prompt batching | Additional native forward/backward with two prompts; logits `[2,512,512]`, finite decoder gradients, no image-encoder gradients |
| Validation search | Nine settings evaluated using cached real YOLO + SAM predictions |
| Submission | Two real test sources; 155 nonempty instance rows, native COCO RLE round trips checked |
| Inference recovery | Rebuilt the same CSV using journals with an inference function that raises if called; CSV hash unchanged |
| Stale thresholds | Mismatched prediction/checkpoint identity rejected |
| Edge prompts | Extreme positive jitter at the lower/right image edge retains valid positive-width/height boxes |

That earlier workflow ran before small plotting, annotation-check and edge-box
improvements. Those changes were checked separately against that earlier notebook,
including a real two-prompt native SAM forward/backward and journal recovery.

## Data and metric contracts

The notebook's own checks cover:

- Union across overlapping masks and a separate component, with exact foreground recovery.
- 4- versus 8-connectivity at diagonal contact.
- COCO polygon, compressed RLE and uncompressed RLE decoding.
- Fortran-order RLE orientation and an exact round trip for a long, two-pixel-wide native mask.
- Rejection of 1024-pixel model input.
- Strict PQ threshold: exactly 0.5 IoU does not match.
- Multiple overlapping GT instances matching one prediction.
- Empty GT, empty predictions and all-empty aggregation.
- Source stems containing underscores and omission of empty submission instances.

In addition, 50 randomized overlap cases were compared with the original
`get_overlap_matrices`, `fp_count_hit` and `fn_count_hit` functions extracted from
the supplied organizer notebook. TP/FP/FN and matched-IoU sums agreed, including
empty cases. These checks do not import the custom U-Net implementation.

## Environment and artifacts

The completed smoke run used Python 3.12.10, PyTorch `2.14.1+cu132`, torchvision
`0.29.1+cu132`, Ultralytics `8.4.198`, NumPy `2.5.3`, SciPy `1.18.1`, and
pycocotools `2.0.11`. The original local torchvision build lacked CUDA NMS; it was
replaced with its matching CUDA build. The notebook now checks that operator
before data preparation.

The ignored local `outputs/` directory contains `executed_smoke.ipynb` and
`solar_yolo_sam_smoke/` with checkpoints, reports, figures, caches, journals and
`smoke_submission.csv`. Pretrained weights and generated artifacts are excluded
from version control. The existing `custom-Unet` folder was not edited.

The one-epoch smoke validation PQ was **0.0**. The deliberately short run verifies
data flow and training execution; its detector produces many false positives and
has not learned the competition task. Do not submit the smoke CSV or interpret
its 155 predicted masks as evidence of useful accuracy. Full-run quality, runtime,
other GPUs, Linux/Kaggle execution and multi-GPU training were not measured here.
