"""Behavioral checks for the v3 notebook, metric, supervision and resource plan."""
import ast
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import nbformat
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("filament_v3_test_runtime", HERE/"pq50_v3_runtime.py")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


def test_owner_map_keeps_instance_identity_and_background_collar():
    a, b = np.zeros((48, 64), bool), np.zeros((48, 64), bool)
    a[5:9, 5:12] = True
    a[15:18, 5:8] = True  # disconnected parts of ONE annotated filament
    b[28:34, 40:55] = True
    owner = runtime.make_owner_map([runtime.encode(a), runtime.encode(b)], radius=4, shape=a.shape)
    assert np.all(owner[a] == 1)
    assert np.all(owner[b] == 2)
    assert owner[5, 4] == 1
    assert owner[0, 63] == 0
    assert owner.dtype == np.uint16


def test_instance_loss_weights_filaments_equally_despite_area():
    fg = torch.zeros((1, 1, 24, 24))
    owner = torch.zeros_like(fg)
    fg[:, :, 1:3, 1:3] = 1
    owner[:, :, 1:3, 1:3] = 1
    fg[:, :, 8:18, 8:18] = 1
    owner[:, :, 8:18, 8:18] = 2
    p = torch.zeros_like(fg)
    p[:, :, 1:3, 1:3] = .6
    p[:, :, 8:18, 8:18] = .2
    expected = ((1-(4*.6+1)/5) + (1-(100*.2+1)/101))/2
    assert runtime.instance_iou_loss(p, fg, owner).item() == pytest.approx(expected)


def test_instance_loss_penalizes_background_and_has_finite_gradients():
    fg, owner = torch.zeros((2, 1, 32, 32)), torch.zeros((2, 1, 32, 32))
    fg[0, :, 10:15, 8:20] = 1
    owner[0, :, 5:20, 3:25] = 1
    p = fg*.8
    base = runtime.instance_iou_loss(p, fg, owner)
    p[0, :, 5:10, 3:25] = .8
    assert runtime.instance_iou_loss(p, fg, owner) > base
    logits = torch.randn((2, 2, 32, 32), requires_grad=True)*40
    logits.retain_grad()
    target = torch.cat([fg, torch.zeros_like(fg), owner], 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = runtime.dense_loss(logits, target)
    loss.backward()
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_metric_matches_organizer_strict_half_iou():
    gt = np.zeros((8, 8), bool)
    gt[1:3, 1:5] = True
    half = np.zeros_like(gt)
    half[1:3, 1:3] = True
    assert runtime.summarize(runtime.metric_counts([runtime.encode(gt)], [runtime.encode(gt)]))["PQ"] == 1
    counts = runtime.metric_counts([runtime.encode(gt)], [runtime.encode(half)])
    assert list(counts) == [0, 0, 1, 1]


def test_crop_dataset_epoch_augmentation_and_id_integrity(tmp_path):
    image = np.zeros((2048, 2048, 3), np.uint8)+130
    packed = np.zeros((2048, 2048), np.uint8)
    owner = np.zeros((2048, 2048), np.uint16)
    packed[900:950, 900:1000] = 1
    owner[884:966, 884:1016] = 3
    cv2.imwrite(str(tmp_path/"image.png"), image)
    cv2.imwrite(str(tmp_path/"packed.png"), packed)
    cv2.imwrite(str(tmp_path/"owner.png"), owner)
    source = dict(cache=str(tmp_path/"image.png"), views=[dict(id="a", boxes=[[900, 900, 100, 50]],
        target=str(tmp_path/"packed.png"), owner=str(tmp_path/"owner.png"))])
    ds = runtime.CropDataset([source], 128, 12, 111)
    x, y = ds[0]
    x2, y2 = ds[0]
    assert torch.equal(x, x2) and torch.equal(y, y2)
    assert x.shape == y.shape == (3, 128, 128)
    assert set(y[2].unique().tolist()) <= {0., 3.}
    assert y[0].sum() > 0
    ds.epoch = 1
    x3, y3 = ds[0]
    assert not torch.equal(x, x3) or not torch.equal(y, y3)


def test_budget_reduces_steps_for_slower_hardware():
    cfg = runtime.default_config()
    fast = runtime.plan_steps(cfg, .1, 0, 500)
    slow = runtime.plan_steps(cfg, 2., 0, 500)
    assert 8 <= slow < fast <= cfg["dense_steps"]
    cfg["session_started"] -= 10*3600
    with pytest.raises(TimeoutError):
        runtime.check_deadline(cfg, "test", reserve=1)


def test_candidate_calibration_variants_and_test_variant_consistent(tmp_path):
    image = np.full((2048, 2048, 3), 130, np.uint8)
    cv2.imwrite(str(tmp_path/"image.png"), image)
    maps = np.zeros((2, 2, 2048, 2048), np.uint8)
    maps[0, 0, 500:510, 500:530] = 240
    maps[1, 0, 1000:1010, 1000:1030] = 240
    source = dict(cache=str(tmp_path/"image.png"))
    np.savez_compressed(tmp_path/"cal.npz", schema=np.array(3), prob=maps.mean(0).astype(np.uint8),
        member_ids=np.array([0, 1]), member_prob=maps, variant=np.array("avg"))
    sets = runtime.candidate_sets(source, tmp_path/"cal.npz", [.35])
    assert set(sets) == {f"{v}_{f}_0.35" for v in ("avg", "v0", "v1") for f in ("dense", "close")}
    assert len(sets["avg_dense_0.35"]) == 2
    assert len(sets["v0_dense_0.35"]) == 1
    np.savez_compressed(tmp_path/"test.npz", schema=np.array(3), prob=maps[0],
        member_ids=np.array([0]), member_prob=np.empty(0, np.uint8), variant=np.array("v0"))
    final = runtime.candidate_sets(source, tmp_path/"test.npz", [.35])
    assert set(final) == {"v0_dense_0.35", "v0_close_0.35"}
    recipe = dict(set="v0_dense_0.35", quality=False, cutoff=.5, min_area=40)
    selected = runtime.select_instances(final[recipe["set"]], recipe)
    assert selected == [final[recipe["set"]][0]["rle"]]


def test_notebook_self_contained_compilable_and_gpu_enabled():
    nb = nbformat.read(HERE/"solar-filament-instance-balanced-under-10h.ipynb", as_version=4)
    nbformat.validate(nb)
    sections = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        if source.startswith("%%filament_section"):
            source = source.split("\n", 1)[1]
            sections.append(source)
        compile(source, "notebook-cell", "exec")
        assert cell.execution_count is None and not cell.outputs
    worker = "\n\n".join(sections)+'\n\nif __name__ == "__main__":\n    main()\n'
    compile(worker, "embedded-worker", "exec")
    assert ast.dump(ast.parse(worker)) == ast.dump(ast.parse((HERE/"pq50_v3_runtime.py").read_text(encoding="utf8")))
    assert nb.metadata.kaggle.isGpuEnabled is True
    assert "from ultralytics" not in worker
    assert "def audit_split_leakage" in worker
    assert "instance_iou_loss" in worker
    assert len(worker) > 60000


def test_warm_start_rejects_changed_heldout_split(tmp_path):
    splits = {k: [dict(stem=k+"_image", image_sha="same-hash")] for k in ("train", "calibration", "report")}
    previous = dict(config=dict(annotation_sha="annotation", fingerprint="previous"), splits=splits)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(previous))
    cfg = dict(warm_start_checkpoints=["unused.pt"], previous_manifest=str(path),
               warm_start_sha256=["checkpoint-hash"], annotation_sha="annotation", run=str(tmp_path))
    runtime.verify_previous_split(cfg, dict(splits=splits))
    changed = json.loads(json.dumps(splits))
    changed["report"][0]["image_sha"] = "different-image"
    with pytest.raises(ValueError, match="report"):
        runtime.verify_previous_split(cfg, dict(splits=changed))


def test_zero_initialized_refinement_preserves_warm_start_prediction():
    model = runtime.build_dense(dict(encoder="resnet18", crop=64, refine=True), pretrained=False).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.equal(model.core(x), model(x))


def test_component_tree_selects_whole_parent_or_disjoint_children():
    parent = np.zeros((32, 32), bool)
    parent[5:20, 5:25] = True
    left, right = np.zeros_like(parent), np.zeros_like(parent)
    left[6:18, 6:12] = True
    right[6:18, 17:23] = True
    candidates = [dict(rle=runtime.encode(mask), area=int(mask.sum()), quality_oof=score)
                  for mask, score in [(parent, .4), (left, .36), (right, .36)]]
    forest = runtime.ComponentTree(candidates+candidates[:1])
    assert len(forest) == 3  # duplicate masks must not receive repeated credit
    recipe = dict(set="v0_dense_tree", quality=True, cutoff=.2, min_area=1)
    selected = runtime.select_instances(forest, recipe, oof=True)
    assert len(selected) == 2
    assert not np.any(runtime.decode(selected[0]) & runtime.decode(selected[1]))
    forest = runtime.ComponentTree([dict(c, quality_oof=.75 if i==0 else .3) for i,c in enumerate(candidates)])
    assert runtime.select_instances(forest, recipe, oof=True) == [runtime.encode(parent)]


def test_stability_choice_prefers_average_for_an_uncertain_tiny_gain():
    candidates = [dict(set="v1_dense_0.35"), dict(set="avg_dense_0.35")]
    # Alternating date advantages make the tiny pooled single-model edge uncertain.
    dates = {str(i): np.asarray([[3.6+(.3 if i%2 else -.29), 6, 3, 4], [3.6, 6, 3, 4]]) for i in range(10)}
    counts = sum(dates.values())
    selected, audit = runtime.stable_recipe_choice(runtime.default_config(), candidates, counts, dates)
    assert selected == 1 and audit[0]["chose_alternative"]
    assert audit[0]["date_paired_95ci"][0] < 0 < audit[0]["date_paired_95ci"][1]


def test_stability_choice_keeps_a_reliably_better_single_model():
    candidates = [dict(set="v1_dense_0.35"), dict(set="avg_dense_0.35")]
    dates = {str(i): np.asarray([[5.0, 7, 2, 3], [3., 5, 4, 5]]) for i in range(10)}
    selected, audit = runtime.stable_recipe_choice(runtime.default_config(), candidates, sum(dates.values()), dates)
    assert selected == 0 and not audit[0]["chose_alternative"]


@pytest.mark.parametrize("encoder", ["resnet34", "tu-tf_efficientnetv2_s"])
def test_real_model_forward_backward_finite_cpu(encoder):
    model = runtime.build_dense(dict(encoder=encoder, crop=128, refine=True), pretrained=False)
    runtime.train_mode(model)
    x = torch.randn(1, 3, 128, 128)
    y = torch.zeros(1, 3, 128, 128)
    y[:, 0, 30:40, 30:90] = 1
    y[:, 2, 20:50, 20:100] = 1
    loss = runtime.dense_loss(model(x), y)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
