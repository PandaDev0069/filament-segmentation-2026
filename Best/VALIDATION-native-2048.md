# Native YOLO26 validation evidence

Validated locally on November 2, 2026 with Python 3.12.10, PyTorch 2.14.1+cu132,
Ultralytics 8.4.198 and one NVIDIA GeForce RTX 3070 Ti Laptop GPU with 8 GiB VRAM.
The default notebook targets Kaggle T4 ×2. Direct local verification used one GPU;
the user also supplied a successful raw two-T4 smoke log, described below.

`python -m pytest Best/test_native_yolo.py -q` passed **10 tests**. They check:

- Notebook schema, Python compilation and equality of embedded/runtime source.
- Native union across masks, one-pixel/thin-component preservation and exact RLE round trips.
- All four preprocessing variants and native dataset flips with `cv2.resize` forbidden.
- A one-pixel target's contribution to mask loss and finite, nonzero gradients.
- Strict PQ threshold, overlapping original GT instances and empty predictions.
- Actual dataset grouping: 707 physical sources, 1,154 annotator records, disjoint
  pixel hashes across partitions and exact union reconstruction for a multi-annotator source.
- Mocked two-GPU launch scheduling: complete/disjoint inference shards, isolated
  visible devices and a two-rank torchrun training command. This is a scheduling
  contract check, not a distributed hardware test.
- Nonempty submission RLE, single-pixel output preservation and zero-detection source coverage.
- Weight resolution completes before launch, downloads only once, and supplies an
  absolute checkpoint path plus SHA-256 to the workers.
- Ray Tune tracking is disabled and the current `quantize` configuration accepts FP16/FP32.

The full notebook executed all **9 code cells** with local overrides `gpu_ids=[0]`,
absolute dataset/weight paths, and a separate validation output directory.
Raw, CLAHE, percentile contrast and gamma each completed one epoch on two native
training sources and two native validation sources. All recorded train/validation
losses were finite. Each produced `best.pt`, `last.pt`, history, training plots,
native tensor evidence, validation caches and threshold metrics. Tensor audits show
image shape `[1,3,2048,2048]` and mask shape `[1,2048,2048]`.

The notebook also completed variant comparison, artifact display, validation
visualization, test prediction and smoke CSV/coverage-manifest generation. All
one-epoch models returned zero validation detections, so their PQ values of zero
are execution evidence only. The selected variant in that tie was CLAHE; this does
not establish that CLAHE is better. The two smoke test sources were recorded in
the manifest even though the CSV had no instance rows.

To exercise real nonempty predictions, the preserved best-run `best.pt` was loaded
through the new native tensor inference path. It returned 6 and 3 YOLO instances
on the two validation probe sources, and 0 and 2 on the two test probe sources.
Every cached mask decoded to nonempty 2048 × 2048 pixels. Threshold scoring and
the submission encoder completed successfully. These are compatibility checks;
the old checkpoint may have trained on the probe sources, so their scores are not
an independent benchmark of this new pipeline.

Local evidence:

- [Executed smoke notebook](validation/executed_native_smoke.ipynb)
- [Machine-readable verification report](validation/validation_report.json)
- [Four-variant smoke outputs](validation/notebook_smoke_smoke/)
- [Saved-checkpoint nonempty inference probe](validation/baseline_inference/)

The user-provided Kaggle log reports Python 3.13.15, PyTorch 2.11.0+cu128,
Ultralytics 8.4.198 and two Tesla T4s. The raw smoke training stage completed one
epoch at `imgsz=2048`, saved both checkpoints, and both validation workers completed
their disjoint source shards. The reported training memory was 5.26 GiB. Both
validation predictions were empty, consistent with the local one-epoch smoke.
This is supplied log evidence; the Kaggle artifacts and per-rank tensor audits
have not been inspected directly.

That log also exposed concurrent pretrained downloads and nonfatal Ray Tune/
deprecated `half` messages. The updated runtime resolves weights before spawning
DDP, disables Ray Tune integration, and uses `quantize=16` for FP16 prediction.
The training algorithm and native-grid policy are unchanged by this cleanup.

Full-data training, a complete four-variant dual-T4 workflow, throughput and new
competition quality remain unmeasured. Let all smoke stages finish before full
mode, and inspect both `native_tensor_rank0.json` and `native_tensor_rank1.json`.
