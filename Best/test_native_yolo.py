"""Native-grid contracts and real-data preparation checks (no model training)."""
import ast
import json
from pathlib import Path

import cv2
import nbformat
import numpy as np
import pytest
import torch

import native_yolo_runtime as native

HERE = Path(__file__).resolve().parent


def notebook_config():
    nb = nbformat.read(HERE / "solar-filament-yolo26-native-2048.ipynb", as_version=4)
    ns = {}
    exec(nb.cells[1].source, ns)
    return ns["CFG"]


def test_notebook_compiles_and_embeds_current_runtime():
    nb = nbformat.read(HERE / "solar-filament-yolo26-native-2048.ipynb", as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell-{i}", "exec")
    embedded = next(c.source for c in nb.cells if c.cell_type == "code" and c.source.startswith("RUNTIME_SOURCE"))
    source = ast.literal_eval(ast.parse(embedded).body[0].value)
    assert source.strip() == (HERE / "native_yolo_runtime.py").read_text(encoding="utf-8").strip()


def test_native_merge_preserves_thin_components_and_rle():
    a = np.zeros((2048, 2048), bool)
    b = a.copy()
    a[10:16, 20:30] = True
    b[14:19, 27:35] = True
    b[2000, 2000] = True
    b[1000, 1000:1005] = True
    labels, boxes = native.union_labels([a, b])
    assert labels.max() == 3
    assert np.array_equal(labels > 0, a | b)
    assert min(box[2] for box in boxes) == 1 / 2048
    assert np.array_equal(native.decode_mask(native.encode_mask(labels > 0)), a | b)
    with pytest.raises(ValueError, match="Resizing is forbidden"):
        native.union_labels([a[::2, ::2]])


def test_all_preprocessing_and_dataset_flips_without_resampling(tmp_path, monkeypatch):
    cfg = notebook_config()
    cfg.update(fliplr=1.0, flipud=1.0, brightness=0.0)
    image = np.full((2048, 2048), 70, np.uint8)
    image[500:1500, 300:1700] = np.arange(1400, dtype=np.uint16)[None] % 200 + 20
    labels = np.zeros_like(image, dtype=np.int32)
    labels[0, 0] = 1
    image_path, target_path = tmp_path / "image.png", tmp_path / "mask.npz"
    cv2.imwrite(str(image_path), image)
    np.savez_compressed(target_path, labels=labels)

    def reject_resize(*args, **kwargs):
        raise AssertionError("Image or target resampling attempted")

    monkeypatch.setattr(cv2, "resize", reject_resize)
    for variant in ("raw", "clahe", "percentile", "gamma"):
        processed = native.preprocess(image, variant, cfg["preprocessing"])
        assert processed.shape == image.shape and processed.dtype == image.dtype
    assert np.array_equal(native.preprocess(image, "raw", cfg["preprocessing"]), image)
    record = {"path": str(image_path), "target": str(target_path), "boxes": [[0.5/2048, 0.5/2048, 1/2048, 1/2048]]}
    sample = native.NativeDataset([record], cfg, "raw", training=True)[0]
    assert sample["masks"][0, -1, -1] == 1
    assert sample["masks"].sum() == 1
    assert np.array_equal(sample["img"][0].numpy(), np.flip(image, (0, 1)))
    batch = native.NativeDataset.collate_fn([sample, sample])
    assert batch["img"].shape == (2, 3, 2048, 2048)
    assert batch["masks"].shape == (2, 2048, 2048)
    assert batch["batch_idx"].tolist() == [0, 1]


def test_native_loss_sees_single_pixel_and_has_finite_gradients():
    proto = torch.zeros((2, 8, 8), requires_grad=True)
    coefficient = torch.ones(2, requires_grad=True)
    labels = torch.zeros((2048, 2048), dtype=torch.int32)
    labels[123, 456] = 1
    original = labels.clone()
    loss = native.native_anchor_loss(coefficient, proto, labels, torch.tensor(1),
                                    torch.tensor([456., 123., 457., 124.]), torch.tensor(1 / 2048**2))
    assert loss.item() == pytest.approx(np.log(2), rel=1e-5)
    loss.backward()
    assert torch.isfinite(proto.grad).all() and proto.grad.abs().sum() > 0
    assert torch.equal(labels, original)


def test_two_image_native_loss_preserves_each_images_target_and_gradients():
    from types import SimpleNamespace
    loss_fn = native.NativeMaskLoss.__new__(native.NativeMaskLoss)
    loss_fn.hyp = SimpleNamespace(box=1.0)
    loss_fn.loss_names = ("box", "seg", "cls", "dfl", "sem")
    fg = torch.ones((2, 1), dtype=torch.bool)
    indices = torch.zeros((2, 1), dtype=torch.long)
    boxes = torch.tensor([[[0., 0., 1., 1.]], [[2047., 2047., 2048., 2048.]]])
    loss_fn.get_assigned_targets_and_loss = lambda preds, batch: ((fg, indices, boxes, None, None), torch.zeros(3), {})
    proto = torch.zeros((2, 2, 8, 8), requires_grad=True)
    coefficient = torch.ones((2, 2, 1), requires_grad=True)
    masks = torch.zeros((2, 2048, 2048), dtype=torch.int32)
    masks[0, 0, 0] = 1
    masks[1, 2047, 2047] = 1
    original = masks.clone()
    loss, metrics = loss_fn.loss({"proto": proto, "mask_coefficient": coefficient}, {"masks": masks})
    assert loss[1].item() == pytest.approx(2 * np.log(2), rel=1e-5)
    loss.sum().backward()
    assert torch.isfinite(proto.grad).all()
    assert all(proto.grad[i].sum() < 0 for i in range(2))
    assert torch.equal(masks, original)


def test_training_default_uses_four_global_images_on_two_gpus(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cfg = notebook_config()
    cfg["mode"] = "smoke"
    assert cfg["batch_per_gpu"] == 2 and cfg["smoke_sources"] >= 4 * len(cfg["gpu_ids"])
    job_path = tmp_path / "job.json"
    native.save_json(job_path, {"cfg": cfg, "run": str(tmp_path)})
    captured = {}

    class CaptureTrainer:
        def __init__(self, overrides):
            captured.update(overrides)

        def train(self):
            captured["called"] = True

    monkeypatch.setattr(native, "NativeTrainer", CaptureTrainer)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, device: None)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=15 * 2**30))
    native.train_job(job_path)
    assert captured["batch"] == 4 and captured["device"] == "0,1"
    assert captured["imgsz"] == 2048 and captured["mask_ratio"] == 1
    assert captured["workers"] == 2 and captured["nbs"] == 64 and captured["called"]


def test_pq_strict_threshold_overlapping_gt_and_empty():
    a = np.zeros((2048, 2048), bool)
    a[0, 0:4] = True
    half = a.copy(); half[0, 2:4] = False
    full, part = native.encode_mask(a), native.encode_mask(half)
    assert native.instance_metrics([full], [full])["PQ"] == 1
    assert native.instance_metrics([part], [full])["tp"] == 0  # exactly 0.5 does not match
    result = native.instance_metrics([full], [full, full])
    assert result["tp"] == 1 and result["fn"] == 1
    assert native.instance_metrics([], [full])["PQ"] == 0
    assert native.instance_metrics([], [])["PQ"] == 1


def test_real_data_grouping_and_native_targets(tmp_path):
    cfg = notebook_config()
    cfg.update(mode="smoke", smoke_sources=2, data_root=str(HERE.parent / "MAGFilo_1.0_Kaggle_2026"))
    manifest_path = native.prepare_data(cfg, tmp_path)
    manifest = native.load_json(manifest_path)
    train, val = manifest["train"], manifest["val"]
    assert len(train) == len(val) == 2
    assert not ({r["pixel_hash"] for r in train} & {r["pixel_hash"] for r in val})
    merged = next(r for r in train if len(r["image_ids"]) > 1)
    payload = native.load_json(manifest["annotation_path"])
    expected = np.zeros((2048, 2048), bool)
    for annotation in payload["annotations"]:
        if str(annotation["image_id"]) in merged["image_ids"]:
            expected |= native.annotation_mask(annotation)
    with np.load(merged["target"]) as target:
        actual = target["labels"]
    assert np.array_equal(actual > 0, expected)
    assert actual.max() == len(merged["boxes"])
    assert len({r["name"] for r in train + val}) == 4
    report = native.load_json(tmp_path / "data_audit.json")
    assert report["source_images"] == 707 and report["annotator_records"] == 1154


def test_dual_gpu_prediction_shards_are_disjoint_complete_and_isolated(tmp_path, monkeypatch):
    cfg = notebook_config()
    weights = tmp_path / "pretrained.pt"
    weights.write_bytes(b"launch-only checkpoint fixture")
    cfg["weights"] = str(weights)
    records = [{"name": f"image{i}.jpeg", "path": f"/sources/image{i}.jpeg"} for i in range(7)]
    manifest = tmp_path / "manifest.json"
    native.save_json(manifest, {"val": records})
    job_path = tmp_path / "job.json"
    native.save_json(job_path, {"cfg": cfg, "manifest": str(manifest), "run": str(tmp_path), "variant": "raw"})
    captured = []
    monkeypatch.setattr(native, "run_commands", lambda commands, log_dir: captured.extend(commands))
    native.launch_prediction(job_path, HERE / "native_yolo_runtime.py", "val")
    assert len(captured) == 2
    assert {c[2]["CUDA_VISIBLE_DEVICES"] for c in captured} == {"0", "1"}
    shards = [native.load_json(c[1][-1])["records"] for c in captured]
    names = [r["name"] for shard in shards for r in shard]
    assert len(names) == len(set(names)) == len(records)
    assert set(names) == {r["name"] for r in records}
    captured.clear()
    native.launch_training(job_path, HERE / "native_yolo_runtime.py")
    assert "--nproc_per_node=2" in captured[0][1]
    assert captured[0][2]["CUDA_VISIBLE_DEVICES"] == "0,1"
    resolved_job = native.load_json(job_path)
    assert resolved_job["resolved_weights"] == str(weights.resolve())
    assert resolved_job["resolved_weights_sha256"] == native.file_sha(weights)


def test_weights_download_finishes_once_before_gpu_launch(tmp_path, monkeypatch):
    from ultralytics.utils import downloads
    monkeypatch.chdir(tmp_path)
    calls = []

    def download(configured):
        calls.append(configured)
        Path(configured).write_bytes(b"complete download fixture")
        return configured

    monkeypatch.setattr(downloads, "attempt_download_asset", download)
    cfg = notebook_config()
    cfg["weights"] = "pretrained.pt"
    job_path = tmp_path / "job.json"
    native.save_json(job_path, {"cfg": cfg, "run": str(tmp_path), "variant": "raw"})

    def launch(commands, log_dir):
        job = native.load_json(job_path)
        path = Path(job["resolved_weights"])
        assert path.is_absolute() and path.read_bytes() == b"complete download fixture"
        assert job["resolved_weights_sha256"] == native.file_sha(path)

    monkeypatch.setattr(native, "run_commands", launch)
    native.launch_training(job_path, HERE / "native_yolo_runtime.py")
    native.launch_training(job_path, HERE / "native_yolo_runtime.py")
    assert calls == ["pretrained.pt"]


def test_optional_ray_tracking_disabled_and_precision_is_current():
    from ultralytics.cfg import get_cfg
    from ultralytics import settings
    assert settings["raytune"] is False
    # Same FP16/FP32 precision modes as the deprecated half flag, without deprecated aliases.
    assert get_cfg(overrides={"quantize": 16}).quantize == 16
    assert get_cfg(overrides={"quantize": None}).quantize is None


def test_submission_rle_and_zero_detection_coverage(tmp_path):
    cfg = notebook_config()
    cfg.update(mode="smoke")
    records = [{"name": "a.jpeg"}, {"name": "b.jpeg"}]
    manifest = tmp_path / "manifest.json"
    native.save_json(manifest, {"test": records})
    job = {"cfg": cfg, "manifest": str(manifest), "run": str(tmp_path), "variant": "raw"}
    job_path = tmp_path / "job.json"
    native.save_json(job_path, job)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights/best.pt").write_bytes(b"provenance-only fixture")
    mask = np.zeros((2048, 2048), bool)
    mask[2047, 2047] = True
    native.save_json(native.prediction_path(job, "test", records[0]), {"instances": [{"rle": native.encode_mask(mask), "confidence": 0.9}]})
    native.save_json(native.prediction_path(job, "test", records[1]), {"instances": []})
    destination = native.create_submission(job_path, 0.25)
    import csv
    with destination.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert destination.name == "smoke_submission.csv"
    assert len(rows) == 1 and rows[0]["filament_id"] == "a_1"
    assert np.array_equal(native.decode_mask({"size": [2048, 2048], "counts": rows[0]["segmentation_rle"]}), mask)
    assert native.load_json(destination.with_suffix(".manifest.json"))["source_coverage"] == [
        {"name": "a.jpeg", "instances": 1}, {"name": "b.jpeg", "instances": 0}]
