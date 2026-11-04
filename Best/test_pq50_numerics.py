"""Regression checks for the actual NaN-training / six-mask submission failure."""
import copy
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pq50_runtime as r


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_bad_probabilities_cannot_become_silent_zero_masks(value):
    p = np.zeros((2, 16, 16), np.float32)
    p[0, 3, 4] = value
    with pytest.raises(FloatingPointError, match="refusing the uint8 conversion"):
        r.quantize_probabilities(p)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_native_prediction_rejects_nonfinite_logits_before_sigmoid(value):
    class BadModel(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 2, *x.shape[-2:]), value)
    with pytest.raises(FloatingPointError, match="inference logits"):
        r.dense_predict(BadModel(), np.zeros((64, 64, 3), np.uint8), torch.device('cpu'), 64, 16, 1, 1)


def test_checkpoint_rejects_nonfinite_state_and_preserves_last_good_file(tmp_path):
    path = tmp_path / 'last.pt'
    r.checkpoint_save(path, dict(state={'w': torch.ones(2)}, spec={'encoder': 'test'}))
    original = path.read_bytes()
    with pytest.raises(FloatingPointError):
        r.checkpoint_save(path, dict(state={'w': torch.tensor([float('nan')])}))
    assert path.read_bytes() == original


def test_loading_poisoned_checkpoint_is_rejected(tmp_path):
    path = tmp_path / 'poison.pt'
    torch.save(dict(state={'w': torch.tensor([float('inf')])}, spec={}), path)
    with pytest.raises(FloatingPointError):
        r.load_dense_checkpoint(path)


def test_nan_gradient_cannot_update_weights_or_ema():
    model = torch.nn.Linear(2, 2)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    original = [p.detach().clone() for p in model.parameters()]
    for p in model.parameters(): p.grad = torch.full_like(p, float('nan'))
    with pytest.raises(FloatingPointError, match='gradient norm'):
        r.dense_step(model, optimizer, ema, 10, .995)
    assert all(torch.equal(a, b) for a, b in zip(original, model.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(original, ema.parameters()))
    assert not optimizer.state


def test_actual_fp32_update_counts_and_backbone_lr():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(3, 4, 1)
            self.decoder = torch.nn.Conv2d(4, 2, 1)
        def forward(self, x): return self.decoder(self.encoder(x).relu())
    model = Tiny()
    ema = copy.deepcopy(model).requires_grad_(False)
    cfg = r.default_config()
    optimizer = r.dense_optimizer(model, cfg)
    assert [g['lr_scale'] for g in optimizer.param_groups] == [.1, 1.]
    for g in optimizer.param_groups: g['lr'] = cfg['dense_lr'] * g['lr_scale']
    before = model.decoder.weight.detach().clone()
    target = torch.zeros(2, 2, 32, 32)
    target[0, 0, 10:15, 8:20] = 1
    loss = r.dense_loss(model(torch.rand(2, 3, 32, 32)), target)
    loss.backward()
    assert r.dense_step(model, optimizer, ema, 0, .995) == 1
    assert not torch.equal(before, model.decoder.weight)
    r.require_finite_state(ema.state_dict())


def test_dense_defaults_avoid_the_failed_fp16_refit_path():
    cfg = r.default_config()
    assert cfg['dense_precision'] == 'float32'
    assert cfg['refit_full'] is False
    assert cfg['encoder_lr_scale'] == .1
    assert cfg['minimum_report_pq'] > .0034510882912068327


def test_calibration_can_exclude_a_bad_mask_network(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'SIZE', 64)
    cfg = r.default_config()
    cfg.update(run=str(tmp_path), fingerprint='repair_test', pixel_thresholds=[.5])
    mask = np.zeros((64, 64), bool); mask[8:24, 10:26] = True
    source = dict(stem='sample', views=[dict(rles=[r.encode(mask)])])
    image = tmp_path / 'image.png'; cv2.imwrite(str(image), np.full((64, 64, 3), 128, np.uint8))
    source['cache'] = str(image)
    r.save_json(tmp_path / 'manifest.json', dict(splits={'calibration': [source]}))
    good = np.zeros((2, 64, 64), np.float32); good[0, mask] = .9
    bad = np.zeros_like(good)
    np.savez_compressed(tmp_path / 'sample.npz', schema=np.array(2),
        member_ids=np.array([0, 1]), member_prob=np.stack([r.quantize_probabilities(good), r.quantize_probabilities(bad)]),
        prob=r.quantize_probabilities((good+bad)/2), boxes=np.zeros((0, 6), np.float32))
    selection = r.select_dense_members(cfg, tmp_path)
    assert selection['member_ids'] == [0]
    assert selection['calibration_selection']['PQ'] == pytest.approx(1.)
    assert r.selected_dense_ids(cfg) == [0]
    selected = r.candidate_sets(source, tmp_path / 'sample.npz', [.5], member=0)
    diluted = r.candidate_sets(source, tmp_path / 'sample.npz', [.5])
    assert len(selected['dense_0.5']) == 1 and not diluted['dense_0.5']


@pytest.mark.parametrize("pq,allowed", [(.003451088, False), (.4, True)])
def test_collapsed_report_stops_export_before_more_gpu_time(tmp_path, pq, allowed):
    cfg = r.default_config(); cfg.update(run=str(tmp_path), fingerprint='current')
    r.save_json(tmp_path / 'report_metrics.json', dict(PQ=pq, fingerprint='current'))
    if allowed:
        r.require_report_ready(cfg)
    else:
        with pytest.raises(RuntimeError, match='collapse guard'):
            r.require_report_ready(cfg)


def test_wrong_run_report_cannot_authorize_export(tmp_path):
    cfg = r.default_config(); cfg.update(run=str(tmp_path), fingerprint='current')
    r.save_json(tmp_path / 'report_metrics.json', dict(PQ=.9, fingerprint='different'))
    with pytest.raises(ValueError, match='different run'):
        r.require_report_ready(cfg)
