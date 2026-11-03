"""Focused checks of the self-contained YOLO preprocessing notebook.

Run with the notebook's pinned Ultralytics version. An optional
FILAMENT_ULTRALYTICS_PATH can point to an isolated pip --target installation.
No training, pretrained-weight download or competition submission is performed.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

if os.environ.get("FILAMENT_ULTRALYTICS_PATH"):
    sys.path.insert(0, os.environ["FILAMENT_ULTRALYTICS_PATH"])

NOTEBOOK = Path(__file__).with_name("solar-filament-yolo26-full-res-0-36-preprocessing.ipynb")


@pytest.fixture(scope="module")
def runtime():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"__name__": "__notebook_test__", "SUBMISSION_PATH": Path("unused_test_submission.csv")}
    for index in [6, 8, 9, 11, 13, 14, 16, 18, 19, 25]:
        source = "".join(notebook["cells"][index]["source"])
        if index in [9, 13, 14, 16, 19, 25]:
            tree = ast.parse(source)
            tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            exec(compile(tree, str(NOTEBOOK), "exec"), ns)
        else:
            exec(compile(source, str(NOTEBOOK), "exec"), ns)
    return ns


def write_maps(ns, tmp_path, probability, single=None, salience=None, disk=None):
    if probability.ndim == 2:
        probability = probability[None]
    path = tmp_path / "maps.npz"
    single = (probability > 127).astype(np.uint8) if single is None else single
    salience = np.zeros(probability.shape[-2:], np.uint8) if salience is None else salience
    np.savez_compressed(path, probability=probability,
                        single_binary=np.packbits(single, axis=-1), salience=salience)
    return dict(file_name="test.jpeg", confidences=[0.1], disk=disk, map_path=str(path))


def test_notebook_schema_and_syntax():
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(nb)
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
            if not cell.source.lstrip().startswith("%pip"):
                ast.parse(cell.source, filename=f"cell_{index}")


def test_every_d4_view_inverts_exactly(runtime):
    ns = runtime
    image = np.arange(49, dtype=np.uint8).reshape(7, 7)
    for k, flip in ns["D4"]:
        transformed = ns["d4_transform"](image, k, flip)
        restored = ns["d4_inverse_maps"](transformed[None], k, flip)[0]
        np.testing.assert_array_equal(restored, image)


@pytest.mark.parametrize("architecture", ["yolo11s-seg.yaml", "yolo26s-seg.yaml"])
def test_model_families_use_the_shared_probability_predictor(runtime, tmp_path, monkeypatch, architecture):
    """Exercise each real model's output format, NMS and prototype decoding without downloads."""
    ns = runtime
    settings = dict(IMAGE_SIZE=128, INFERENCE_DEVICES=["cpu"], GPU_IDS=[], TTA_VIEWS=1,
                    MAX_DETECTIONS=4, CONFIDENCE_CANDIDATES=[1e-8, 1e-5])
    for key, value in settings.items():
        monkeypatch.setitem(ns, key, value)
    model = ns["YOLO"](architecture)
    assert model.task == "segment"
    if architecture.startswith("yolo11"):
        assert type(model.model.model[-1]).__name__ == "Segment"
        assert not getattr(model.model, "end2end", False)
    checkpoint = tmp_path / (Path(architecture).stem + ".pt")
    model.save(checkpoint)
    del model
    with ns["InferenceEngine"](checkpoint) as engine:
        probability, binary = engine.predict_maps(np.full((128, 128), 128, np.uint8))
    assert probability.shape == binary.shape == (2, 128, 128)
    assert probability.dtype == binary.dtype == np.uint8
    assert probability.max() > 0
    assert np.all(probability[1] <= probability[0])


def test_soft_decoder_matches_stock_native_binary_and_confidence(runtime):
    ns = runtime
    torch = ns["torch"]
    torch.manual_seed(37)
    proto = torch.randn(4, 16, 16)
    coefficients = torch.randn(3, 4)
    boxes = torch.tensor([[2., 3., 42., 51.], [12., 10., 63., 62.], [1., 1., 22., 25.]])
    pred = torch.cat([boxes, torch.tensor([[0.8, 0.], [0.3, 0.], [0.08, 0.]]), coefficients], dim=1)
    predictor = object.__new__(ns["NativeProbabilityPredictor"])
    predictor.model = SimpleNamespace(names={0: "filament"})
    with torch.inference_mode():
        result = predictor.construct_result(pred.clone(), torch.zeros(1, 3, 64, 64),
                                            np.zeros((64, 64, 3), np.uint8), "test", proto)
        stock = ns["ops"].process_mask_native(proto, coefficients, boxes, (64, 64))
    for i, threshold in enumerate(ns["CONFIDENCE_CANDIDATES"]):
        selected = pred[:, 4] >= threshold
        expected = stock[selected].amax(0).numpy() if selected.any() else np.zeros((64, 64), np.uint8)
        np.testing.assert_array_equal(result.single_binary[i], expected)
    assert np.all(result.probability_maps[1:] <= result.probability_maps[:-1])
    assert len(np.unique(result.probability_maps)) > 20  # Actual probabilities, not binary/confidence proxies.
    assert not result.probability_maps[:, 0, :].any()  # Sigmoid must not turn outside-box zeros into 0.5.
    with torch.inference_mode():
        empty = predictor.construct_result(pred[:0], torch.zeros(1, 3, 64, 64),
                                           np.zeros((64, 64, 3), np.uint8), "empty", proto)
    assert not empty.probability_maps.any() and not empty.single_binary.any()


def test_solar_transform_native_geometry_and_failed_fit_fallback(runtime):
    ns = runtime
    size = 512
    yy, xx = np.ogrid[:size, :size]
    radius = np.sqrt((yy - 255.5)**2 + (xx - 255.5)**2)
    image = np.where(radius < 230, 180 - 35 * (radius / 230)**2, 0).astype(np.uint8)
    image[200:208, 150:350] = 80
    original = image.copy()
    assert ns["estimate_disk"](image) is not None
    processed = ns["preprocess_grayscale"](image)
    assert processed.shape == image.shape and processed.dtype == np.uint8
    assert processed.flags.c_contiguous and not np.array_equal(processed, image)
    np.testing.assert_array_equal(image, original)
    np.testing.assert_array_equal(processed[:10, :10], image[:10, :10])
    for degenerate in [np.zeros((512, 512), np.uint8), np.full((512, 512), 128, np.uint8)]:
        assert ns["estimate_disk"](degenerate) is None
        np.testing.assert_array_equal(ns["preprocess_grayscale"](degenerate), degenerate)


def test_hysteresis_peak_mean_area_and_disjoint_masks(runtime, tmp_path):
    ns = runtime
    p = np.zeros((64, 64), np.uint8)
    p[5:15, 5:20] = 200
    p[8, 8] = 250  # Good: enough area, mean and a high peak.
    p[25:35, 5:20] = 130
    p[28, 8] = 250  # High peak, low mean: must be rejected.
    p[45:55, 5:20] = 200  # No strong core: must be rejected.
    p[20, 40] = 255  # Too small: must be rejected.
    cached = write_maps(ns, tmp_path, p)
    cfg = dict(confidence=0.1, mode="hysteresis", clip_to_disk=False,
               t_lo=0.35, t_hi=0.9, min_meanp=0.7, min_area=50)
    rles = ns["final_prediction"](cached, cfg)
    assert len(rles) == 1 and int(ns["mask_utils"].area(rles[0])) == 150
    baseline = dict(confidence=0.1, mode="union_cc", clip_to_disk=False, min_area=1)
    rles = ns["final_prediction"](cached, baseline)
    assert len(rles) == 4
    occupied = np.zeros(p.shape, np.uint8)
    for rle in rles:
        mask = ns["decode_mask"](rle)
        assert not occupied[mask].any()
        occupied |= mask


def test_morphology_bridges_supported_gap_without_remote_predictions(runtime, tmp_path):
    ns = runtime
    p = np.zeros((64, 64), np.uint8)
    p[20:27, 10:20] = 240
    p[20:27, 21:31] = 240
    p[20:27, 20] = 70  # Below low threshold; above the restricted p>0.15 support.
    salience = np.zeros_like(p)
    salience[20:27, 20] = 96
    salience[45:50, 45:50] = 255  # Remote dark feature with no YOLO support.
    cached = write_maps(ns, tmp_path, p, salience=salience,
                        disk=dict(cx=32, cy=32, radius=31))
    cfg = dict(confidence=0.1, mode="hysteresis_morph", clip_to_disk=False,
               t_lo=0.35, t_hi=0.8, min_meanp=0.5, min_area=20, bridge_px=3)
    rles = ns["final_prediction"](cached, cfg)
    assert len(rles) == 1
    mask = ns["decode_mask"](rles[0])
    assert mask[23, 20] and not mask[45:, 45:].any()


def test_cache_avoids_inference_and_invalidates_changed_source(runtime, tmp_path):
    ns = runtime
    path = tmp_path / "disk.jpeg"
    image = np.zeros((2048, 2048), np.uint8)
    ns["cv2"].imwrite(str(path), image)
    cache = tmp_path / "cache"
    cache.mkdir()
    class FakeEngine:
        calls = 0
        def predict_maps(self, processed):
            self.calls += 1
            assert processed.shape == (2048, 2048)
            shape = (len(ns["CONFIDENCE_CANDIDATES"]), 2048, 2048)
            return np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    engine = FakeEngine()
    first = ns["cached_predict"](engine, path, cache)
    hit = ns["cached_predict"](None, path, cache)
    assert first == hit and engine.calls == 1
    maps = ns["load_maps"](first)
    assert maps["single_binary"].shape == maps["probability"].shape
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(RuntimeError, match="stale"):
        ns["cached_predict"](None, path, cache)


def test_bounded_pool_caps_tasks_and_propagates_failures(runtime):
    ns = runtime
    lock = threading.Lock()
    active = peak = 0
    def task(i):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return i * 2
    assert list(ns["bounded_map"](task, range(12), workers=4)) == list(range(0, 24, 2))
    assert 1 < peak <= 4
    def fail(i):
        if i == 2:
            raise ValueError("worker failure")
        return i
    with pytest.raises(ValueError, match="worker failure"):
        list(ns["bounded_map"](fail, range(8), workers=4))


def test_gpu_lanes_are_exclusive_and_inverse_average(runtime):
    ns = runtime
    active, seen = [0, 0], [0, 0]
    lock = threading.Lock()
    canonical = np.arange(64, dtype=np.uint8).reshape(8, 8)
    class FakeYOLO:
        def __init__(self, lane):
            self.lane = lane
            self.view = 0
        def predict(self, source, **kwargs):
            with lock:
                active[self.lane] += 1
                assert active[self.lane] == 1
                seen[self.lane] += 1
            k, flip = ns["D4"][self.view % ns["TTA_VIEWS"]]
            self.view += 1
            data = ns["d4_transform"](canonical, k, flip)[None]
            time.sleep(0.002)
            with lock:
                active[self.lane] -= 1
            return [SimpleNamespace(probability_maps=data, single_binary=(data > 31).astype(np.uint8))]
    engine = object.__new__(ns["InferenceEngine"])
    engine.lanes = ns["queue"].Queue()
    for lane in range(2):
        engine.lanes.put((FakeYOLO(lane), ns["torch"].device("cpu")))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(engine.predict_maps, [canonical] * 4))
    for probability, binary in results:
        # All confidence channels receive the same synthetic map in this mock.
        for probability_map in probability:
            np.testing.assert_array_equal(probability_map, canonical)
        np.testing.assert_array_equal(binary[0], canonical > 31)
    assert all(seen)


def test_validation_workers_disabled_without_changing_training(runtime, monkeypatch):
    ns = runtime
    calls = []
    def get_loader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        calls.append((self.args.workers, batch_size, rank, mode))
        return object()
    monkeypatch.setattr(ns["SegmentationTrainer"], "get_dataloader", get_loader)
    trainer = object.__new__(ns["FourWorkerSegmentationTrainer"])
    trainer.args = SimpleNamespace(workers=2)
    trainer.get_dataloader("dataset", 1, 0, "train")
    trainer.get_dataloader("dataset", 2, 0, "val")
    assert calls == [(2, 1, 0, "train"), (0, 1, 0, "val")]
    assert trainer.args.workers == 2


def test_submission_native_roundtrip_and_overlap_rejection(runtime, tmp_path):
    ns = runtime
    mask = np.zeros((2048, 2048), np.uint8)
    mask[500:510, 500:600] = 1
    rle = ns["encode_mask"](mask)
    rows = ns["submission_rows_for_source"]("20140609195854Bh", [rle])
    path = tmp_path / "submission.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = ns["csv"].writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        writer.writerows(rows)
    audit = ns["audit_submission"](path, {"20140609195854Bh": 1, "empty": 0})
    assert audit["non_overlapping_masks"] and audit["zero_prediction_images"] == 1
    with path.open("a", newline="", encoding="utf-8") as stream:
        ns["csv"].writer(stream).writerow(["20140609195854Bh_2", rows[0][1]])
    with pytest.raises(ValueError, match="overlaps"):
        ns["audit_submission"](path, {"20140609195854Bh": 2})


def test_calibration_report_and_parallel_export_share_frozen_pipeline(runtime, tmp_path, monkeypatch):
    """Execute the selection cell and real CSV exporter with deterministic predictions."""
    ns = runtime
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    test_dir = tmp_path / "test_images"
    test_dir.mkdir()
    stems = ["20140609195854Bh", "20111116063134Lh", "20120512063234Lh", "20120405130014Th"]
    for index, stem in enumerate(stems):
        directory = image_dir if index < 2 else test_dir
        assert ns["cv2"].imwrite(str(directory / (stem + ".jpeg")), np.zeros((2048, 2048), np.uint8))
    gt = np.zeros((2048, 2048), np.uint8)
    gt[200:210, 200:250] = 1
    gt_rle = ns["encode_mask"](gt)
    class FakeEngine:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint
        def predict_maps(self, processed):
            prediction = gt.copy()
            if self.checkpoint.stem == "weak":
                prediction[200:210, 250:270] = 1
            elif self.checkpoint.stem == "loser":
                prediction[:] = 0
            binary = np.repeat(prediction[None], len(ns["CONFIDENCE_CANDIDATES"]), axis=0)
            return binary * 255, binary
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    checkpoints = [tmp_path / name for name in ["weak.pt", "best.pt", "loser.pt"]]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"synthetic checkpoint; no model loaded: " + checkpoint.name.encode())
    pd = ns["pd"]
    calibration = pd.DataFrame(dict(image_id=[1, 2], file_name=[stems[0] + ".jpeg"] * 2))
    report = pd.DataFrame(dict(image_id=[3], file_name=[stems[1] + ".jpeg"]))
    configs = [dict(confidence=0.1, mode="union_cc", min_area=1, clip_to_disk=False),
               dict(confidence=0.1, mode="hysteresis", t_lo=.35, t_hi=.9,
                    min_meanp=.8, min_area=200, clip_to_disk=False)]
    replacements = dict(InferenceEngine=FakeEngine, EXPERIMENT_DIR=experiment,
                        TRAIN_IMAGE_DIR=image_dir, TEST_IMAGE_DIR=test_dir,
                        RUN_FINGERPRINT="test_training_signature", checkpoint_candidates=checkpoints,
                        PREDICTION_CACHE_POLICY="bounded", PREDICTION_PREVIEW_COUNT=1,
                        calibration_records=calibration, report_records=report,
                        POSTPROCESS_GRID=configs, training_seconds=0,
                        target_rles=lambda image_id: [gt_rle])
    for key, value in replacements.items():
        monkeypatch.setitem(ns, key, value)
    # display is provided by Jupyter during normal execution.
    monkeypatch.setitem(ns, "display", lambda *args: None)
    monkeypatch.setattr(ns["plt"], "show", lambda: None)
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    exec(compile("".join(notebook["cells"][21]["source"]), "selection_cell", "exec"), ns)
    assert float(ns["report_summary"].iloc[0].PQ) == pytest.approx(1.0)
    frozen = json.loads((experiment / "selected_pipeline.json").read_text())
    assert Path(frozen["checkpoint"]).name == "best.pt"
    assert frozen["postprocessing"] == ns["SELECTED_CONFIG"]
    for checkpoint in [checkpoints[0], checkpoints[2]]:
        assert not list(ns["inference_cache_dir"](checkpoint).glob("*.npz"))
        assert checkpoint.is_file()  # Cache pruning must preserve all weight files.
    output = tmp_path / "export.csv"
    audit = ns["create_submission_parallel"](output)
    assert audit["test_images_processed"] == 2 and audit["submitted_filaments"] == 2
    assert audit["pipeline"]["postprocessing"] == frozen["postprocessing"]
    assert audit["non_overlapping_masks"]
    retained = {p.stem for p in ns["SELECTED_CACHE_DIR"].glob("*.npz")}
    assert retained == {stems[0], stems[1], stems[3]}  # Calibration, report preview, first sorted test preview.
    with monkeypatch.context() as changed:
        changed.setitem(ns, "TTA_VIEWS", 1)
        with pytest.raises(ValueError, match="changed after selection"):
            ns["create_submission_parallel"](tmp_path / "changed.csv")


def test_disk_preflight_fits_19_5_gib_and_keeps_real_capacity_guard(runtime, tmp_path, monkeypatch):
    ns = runtime
    pd = ns["pd"]
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    for i in range(180):
        (test_dir / f"{i:03d}.JPEG").touch()
    def frame(count):
        return pd.DataFrame(dict(file_name=[f"{i}.jpeg" for i in range(count)]))
    replacements = dict(EXPERIMENT_DIR=experiment, YOLO_DATA_DIR=experiment / "yolo_filament_dataset",
                        RUNS_DIR=experiment / "yolo_runs", TEST_IMAGE_DIR=test_dir,
                        RUN_FINGERPRINT="same_training", train_records=frame(561),
                        calibration_records=frame(74), report_records=frame(72),
                        PREDICTION_CACHE_POLICY="bounded", PREDICTION_PREVIEW_COUNT=10,
                        NUM_WORKERS=4, RUN_TRAINING=True, NUM_EPOCHS=5, SAVE_EVERY_EPOCH=True,
                        REBUILD_YOLO_DATASET=False)
    for key, value in replacements.items():
        monkeypatch.setitem(ns, key, value)
    disk = SimpleNamespace(total=60 * ns["GIB"], used=40.5 * ns["GIB"], free=19.5 * ns["GIB"])
    monkeypatch.setattr(ns["shutil"], "disk_usage", lambda path: disk)
    monkeypatch.setattr(ns["psutil"], "virtual_memory", lambda: SimpleNamespace(available=20 * ns["GIB"]))
    assert ns["resource_preflight"]() is False  # Disable the optional RGB image cache.
    plan = json.loads((experiment / "resource_plan.json").read_text())
    assert 14 < plan["required_gib"] < 19.5
    assert 4 < plan["prediction_peak_gib"] < 5
    assert plan["peak_cached_images"] == 152
    assert plan["reserve_gib"] == 6  # Fit by changing retention, not reducing the reserve.
    monkeypatch.setitem(ns, "PREDICTION_CACHE_POLICY", "all")
    with pytest.raises(OSError, match="peak disk headroom"):
        ns["resource_preflight"]()
    monkeypatch.setitem(ns, "PREDICTION_CACHE_POLICY", "bounded")
    disk.free = 8 * ns["GIB"]
    with pytest.raises(OSError, match="peak disk headroom"):
        ns["resource_preflight"]()


def test_cache_cleanup_rejects_other_directories_and_preserves_unknown_files(runtime, tmp_path, monkeypatch):
    ns = runtime
    experiment = tmp_path / "experiment"
    root = experiment / "prediction_cache" / ("a" * 16)
    root.mkdir(parents=True)
    monkeypatch.setitem(ns, "EXPERIMENT_DIR", experiment)
    for name in ["loser.npz", "loser.json", "loser.npz.tmp", "keep.npz", "keep.json", "weights.pt"]:
        (root / name).write_bytes(b"keep or remove deliberately")
    with pytest.raises(ValueError, match="Refusing to prune"):
        ns["prune_prediction_cache"](tmp_path)
    assert ns["prune_prediction_cache"](root, keep_stems={"keep"}) > 0
    assert {p.name for p in root.iterdir()} == {"keep.npz", "keep.json", "weights.pt"}
    with pytest.raises(ValueError, match="stem"):
        ns["delete_prediction_source"](root, "../outside")


def test_report_stream_scores_every_image_and_keeps_only_previews(runtime, tmp_path, monkeypatch):
    ns = runtime
    experiment = tmp_path / "experiment"
    root = experiment / "prediction_cache" / ("b" * 16)
    root.mkdir(parents=True)
    foreground = np.zeros((64, 64), np.uint8)
    foreground[10:20, 10:30] = 1
    rle = ns["encode_mask"](foreground)
    def fake_predict(engine, path, cache_dir):
        destination = cache_dir / (path.stem + ".npz")
        np.savez_compressed(destination, probability=foreground[None] * 255,
                            single_binary=np.packbits(foreground[None], axis=-1),
                            salience=np.zeros_like(foreground))
        (cache_dir / (path.stem + ".json")).write_text("{}")
        return dict(map_path=str(destination), confidences=[0.1], disk=None, file_name=path.name)
    for key, value in dict(EXPERIMENT_DIR=experiment, TRAIN_IMAGE_DIR=tmp_path,
                           cached_predict=fake_predict, target_rles=lambda image_id: [rle]).items():
        monkeypatch.setitem(ns, key, value)
    frame = ns["pd"].DataFrame(dict(file_name=["one.jpeg", "two.jpeg", "three.jpeg"], image_id=[1, 2, 3]))
    config = dict(confidence=0.1, mode="union_cc", min_area=1, clip_to_disk=False)
    rows, summary = ns["evaluate_partition"](None, frame, [config], root, "stream test",
                                               retain_cache_stems={"two"})
    assert len(rows) == 3 and summary.iloc[0].PQ == pytest.approx(1.0)
    assert {p.name for p in root.iterdir()} == {"two.npz", "two.json"}
