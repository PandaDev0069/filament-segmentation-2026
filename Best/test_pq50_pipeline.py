"""Behavioral checks for labels, official PQ, native tiling and legal CSV export."""
import copy
import json
from pathlib import Path
import sys

import cv2
import nbformat
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pq50_runtime as r


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(r, "SIZE", 64)
    return 64


def square(x, y, w=8, h=8, shape=(64, 64)):
    value = np.zeros(shape, bool)
    value[y:y+h, x:x+w] = True
    return value


@pytest.mark.parametrize("shape", [(64, 64), (2048, 2048)])
def test_rle_is_column_major_and_roundtrips(shape):
    mask = square(5, 10, 7, 3, shape)
    assert np.array_equal(r.decode(r.encode(mask)), mask)
    assert r.encode(mask)["counts"] == r.encode(np.asfortranarray(mask))["counts"]


@pytest.mark.parametrize("box", [(0, 0, 5, 5), (2043, 2043, 2048, 2048), (10, 0, 13, 2048), (345, 112, 405, 187)])
def test_sparse_roi_rle_equals_native_coco_encoding(box):
    x0, y0, x1, y1 = box
    mask = np.random.default_rng(3).random((y1-y0, x1-x0)) > .4
    mask[-1, -1] = True
    full = np.zeros((2048, 2048), bool)
    full[y0:y1, x0:x1] = mask
    assert r.full_rle(box, mask) == r.encode(full)


@pytest.mark.parametrize("kind,expected", [
    ("perfect", [1., 1, 0, 0]), ("empty_prediction", [0., 0, 0, 1]),
    ("empty_target", [0., 0, 1, 0]), ("both_empty", [0., 0, 0, 0]),
    ("exact_half", [0., 0, 1, 1]), ("duplicate_gt", [2., 2, 0, 0])])
def test_organizer_strict_overlap_all_hits(kind, expected):
    mask = square(3, 4)
    gt, pred = [r.encode(mask)], [r.encode(mask)]
    if kind == "empty_prediction": pred = []
    if kind == "empty_target": gt = []
    if kind == "both_empty": gt, pred = [], []
    if kind == "exact_half": pred = [r.encode(square(3, 4, 16, 8))]
    if kind == "duplicate_gt": gt = gt * 2
    np.testing.assert_allclose(r.metric_counts(gt, pred), expected)


def test_pq_pools_records_not_image_averages():
    a, b = np.array([1., 1, 0, 0]), np.array([0., 0, 0, 9])
    assert r.summarize(a + b)["PQ"] == pytest.approx(1 / 5.5)


def test_polygon_keeps_single_instance_and_disconnected_pieces(small):
    ann = {"segmentation": [[2, 2, 10, 2, 10, 10, 2, 10], [30, 30, 36, 30, 36, 36, 30, 36]]}
    mask = r.decode(r.annotation_rle(ann, (small, small)))
    assert cv2.connectedComponents(mask.astype(np.uint8))[0] == 3


def test_uncertain_geometry_still_normalizes():
    y, x = np.mgrid[:256, :256]
    image = np.clip(20 + .15*x + .1*y, 0, 255).astype(np.uint8)
    features, meta = r.normalize_solar(image)
    assert not meta["geometry_ok"]
    assert features.shape == (256, 256, 3)
    assert features.dtype == np.uint8 and np.isfinite(features).all()
    assert not np.array_equal(features[:, :, 0], image)
    assert features[:, :, 0].std() > 10


@pytest.mark.parametrize("size,tile,overlap", [(2048, 1024, 256), (2048, 512, 64), (64, 64, 8)])
def test_tiling_covers_last_pixel_without_resampling(size, tile, overlap):
    starts = r.tile_starts(size, tile, overlap)
    covered = np.zeros(size, bool)
    for s in starts: covered[s:s + tile] = True
    assert covered.all() and starts[0] == 0 and starts[-1] + tile == size


def test_candidate_guard_keeps_small_confident_components():
    p = np.zeros((64, 64), np.float32)
    p[2:5, 2:5] = .95
    p[20:30, 20:30] = .7
    p[45:50, 45:50] = .8
    kept = list(r.components(p, .5, 3, maximum=2))
    assert len(kept) == 2
    assert sum(mask.sum() for _, mask in kept) == 9 + 25


@pytest.mark.parametrize("tta", [1, 4, 8])
def test_tta_inverse_and_tile_blend_preserve_coordinates(tta):
    class PixelModel(torch.nn.Module):
        def forward(self, x): return x[:, :2]
    image = np.random.default_rng(5).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    pred = r.dense_predict(PixelModel(), image, torch.device("cpu"), tile=64, overlap=16, batch=3, tta=tta)
    values = image[:, :, :2].astype(float) / 255
    expected = 1 / (1 + np.exp(-(values - [.485, .456]) / [.229, .224]))
    np.testing.assert_allclose(pred.transpose(1, 2, 0), expected, atol=2e-6)


def test_small_and_empty_crop_gradients_finite():
    logits = torch.randn((2, 2, 64, 64), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 10:13, 14:17] = 1
    loss = r.dense_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 11, 15] < 0
    assert logits.grad[1, 0, 11, 15] > 0


def candidate(mask, score=.9):
    ys, xs = np.where(mask)
    box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    x0, y0, x1, y1 = box
    features = [0.] * len(r.FEATURES)
    features[1] = score
    return dict(rle=r.encode(mask), local_rle=r.encode(mask[y0:y1, x0:x1]),
        box=box, area=int(mask.sum()), features=features)


def test_exclusive_ownership_deduplicates_preserves_identity(small):
    a = square(5, 5, 15, 5) | square(27, 12, 3, 3)
    b = square(17, 5, 15, 5)
    recipe = dict(quality=False, cutoff=.5, min_area=3)
    output = r.select_instances([candidate(a), candidate(a), candidate(b, .8)], recipe)
    masks = [r.decode(v) for v in output]
    assert len(masks) == 2 and not (masks[0] & masks[1]).any()
    assert np.array_equal(masks[0], a)
    assert cv2.connectedComponents(masks[0].astype(np.uint8))[0] == 3


def test_adaptive_thresholds_choose_quality_and_suppress_nested_duplicates(small):
    class Quality:
        def predict(self, x): return x[:, 0]
    a, b = candidate(square(5, 5, 16, 16)), candidate(square(5, 5, 8, 8))
    a['features'][0], b['features'][0] = .8, .6
    sets = {'dense_0.35': [a], 'dense_0.65': [b], 'hybrid_0.35': [a]}
    values = r.recipe_candidates(sets, 'dense_adaptive')
    assert len(values) == 2
    recipe = dict(set='dense_adaptive', quality=True, cutoff=.2, min_area=3)
    output = r.select_instances(values, recipe, Quality())
    assert len(output) == 1
    assert np.array_equal(r.decode(output[0]), square(5, 5, 16, 16))


def test_quality_target_uses_independent_annotators(small):
    a, b = square(2, 2), square(30, 30)
    source = dict(views=[dict(rles=[r.encode(a)]), dict(rles=[r.encode(b)])])
    assert r.candidate_targets(source, [candidate(a)])[0] == pytest.approx(.5)


def test_unused_detector_refit_can_be_skipped_only_for_semantic_recipe(tmp_path):
    cfg = r.default_config(); cfg['run'] = str(tmp_path)
    r.save_json(tmp_path / 'selection_models' / 'detector' / 'complete.json', dict(checkpoint='selection.pt'))
    r.save_json(tmp_path / 'selected_pipeline.json', dict(recipe=dict(set='dense_0.5')))
    assert r.model_paths(cfg, full=True)['detector'] == 'selection.pt'
    r.save_json(tmp_path / 'selected_pipeline.json', dict(recipe=dict(set='hybrid_0.5')))
    with pytest.raises(RuntimeError, match='full-data detector'):
        r.model_paths(cfg, full=True)


def test_box_fusion_support_counts_views_not_duplicates():
    rows = [[10, 10, 20, 20, .9, 0], [10, 10, 20, 20, .8, 0], [11, 10, 21, 20, .8, 2], [40, 40, 50, 50, .4, 0]]
    boxes = r.fuse_boxes(rows, 4)
    assert boxes.shape == (2, 6)
    assert boxes[0, 5] == pytest.approx(.5)
    assert boxes[1, 5] == pytest.approx(.25)


def test_date_split_before_annotator_expansion(tmp_path, small):
    root = tmp_path / "data"
    (root / "train").mkdir(parents=True)
    (root / "test" / "test_images").mkdir(parents=True)
    images = []
    for date in range(1, 16):
        for site in ("Bh", "Ch"):
            for person in ("a", "b"):
                images.append(dict(id=f"{person}-{date}-{site}", file_name=f"201101{date:02d}120000{site}.jpeg", height=small, width=small))
    data = dict(images=images, annotations=[])
    (root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(data))
    split, _ = r.index_data(root, r.default_config())
    groups = [{s["date"] for s in split[k]} for k in ("train", "calibration", "report")]
    assert all(not groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))
    assert sum(len(s["views"]) for k in split for s in split[k]) == len(images)


def test_preparation_is_idempotent_and_content_fingerprinted(tmp_path, small):
    root = tmp_path / "data"
    images_dir = root / "train" / "train_images"
    tests_dir = root / "test" / "test_images"
    images_dir.mkdir(parents=True); tests_dir.mkdir(parents=True)
    images = []
    gray = np.random.default_rng(2).integers(20, 230, (small, small), dtype=np.uint8)
    for date in range(1, 7):
        name = f"201101{date:02d}120000Bh.jpeg"
        cv2.imwrite(str(images_dir / name), gray)
        images.append(dict(id='a-' + name[:-5], file_name=name, height=small, width=small))
    cv2.imwrite(str(tests_dir / '20120101120000Ch.jpeg'), gray)
    r.save_json(root / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json', dict(images=images, annotations=[]))
    cfg = r.default_config()
    cfg.update(data_root=str(root), output_root=str(tmp_path / 'output'))
    first, _ = r.prepare(cfg)
    second, _ = r.prepare(first)
    assert first['run'] == second['run'] and first['fingerprint'] == second['fingerprint']
    cv2.imwrite(str(tests_dir / '20120101120000Ch.jpeg'), 255-gray)
    changed, _ = r.prepare(cfg)
    assert changed['fingerprint'] != first['fingerprint']


def test_quality_crossfit_never_sees_heldout_date(tmp_path, small, monkeypatch):
    seen, models = [], []
    class SpyEstimator:
        def fit(self, x, y):
            self.seen = set(x[:, 0]); seen.append(self.seen); return self
        def predict(self, x):
            assert not self.seen & set(x[:, 0])
            return np.zeros(len(x))
    monkeypatch.setattr(r, "quality_estimator", lambda seed: SpyEstimator())
    import joblib
    monkeypatch.setattr(joblib, "dump", lambda *args: None)
    sources = [dict(stem=f"s{i}", date=f"d{i}") for i in range(6)]
    r.save_json(tmp_path / "manifest.json", dict(splits=dict(calibration=sources)))
    for i, s in enumerate(sources):
        c = candidate(square(2, 2)); c["features"][0] = i; c["target"] = .6
        r.save_json(tmp_path / (s["stem"] + ".json"), dict(dense_05=[c]))
    cfg = r.default_config(); cfg.update(run=str(tmp_path), quality_folds=3)
    _, data = r.fit_quality(cfg, tmp_path)
    assert len(seen) == 4 and len(seen[-1]) == 6
    assert all("quality_oof" in data[s["stem"]]["dense_05"][0] for s in sources)


def test_submission_validates_all_images_and_native_rles(tmp_path, small, monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *args: None)
    sources = [dict(stem="20110101120000Bh", name="20110101120000Bh.jpeg", photometry=dict(geometry_ok=True)),
               dict(stem="20110102120000Ch", name="20110102120000Ch.jpeg", photometry=dict(geometry_ok=False))]
    r.save_json(tmp_path / "manifest.json", dict(splits=dict(test=sources)))
    r.save_json(tmp_path / (sources[0]["stem"] + ".json"), dict(dense_05=[candidate(square(2, 3))]))
    r.save_json(tmp_path / (sources[1]["stem"] + ".json"), dict(dense_05=[]))
    cfg = r.default_config(); cfg.update(run=str(tmp_path), mode="smoke", annotation_sha="test")
    recipe = dict(set="dense_05", quality=False, cutoff=.5, min_area=3)
    path = r.export_submission(cfg, tmp_path, recipe)
    import pandas as pd
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["filament_id", "segmentation_rle"]
    assert frame.filament_id.tolist() == [sources[0]["stem"] + "_1"]
    audit = r.load_json(tmp_path / "submission_audit.json")
    assert audit["images"] == 2 and audit["zero_prediction_images"] == 1
    assert not (tmp_path / "submission.csv").exists()


def test_notebook_is_standalone_compiles_and_matches_runtime():
    here = Path(__file__).resolve().parent
    nb = nbformat.read(here / "solar-filament-native-hybrid-pq50.ipynb", as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        assert not cell.get("outputs", [])
        if cell.cell_type == "code": compile(cell.source, "notebook_cell", "exec")
    embedded = next(c.source for c in nb.cells if c.cell_type == "code" and "RUNTIME_SOURCE = " in c.source)
    import ast
    value = next(n.value for n in ast.parse(embedded).body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "RUNTIME_SOURCE" for t in n.targets))
    assert ast.literal_eval(value) == (here / "pq50_runtime.py").read_text(encoding="utf8")

