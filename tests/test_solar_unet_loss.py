"""Regression checks for segmentation objectives and validation aggregation.

Run with ``python -m unittest discover -s tests -v``. The distributed check
uses two real CPU/Gloo workers so DDP's gradient averaging is exercised.
"""

import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import solar_unet_runtime as runtime
from solar_unet_runtime import (
    DEFAULT_LOSS_WEIGHTS,
    log_soft_panoptic_quality_loss,
    segmentation_loss,
    summarize_epoch_totals,
)


class SharedLogits(nn.Module):
    """Small shared model whose gradients include both ranks' foreground mix."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([-.6, .3, -.2, .8, -1., .4]).reshape(1, 1, 2, 3))

    def forward(self, offsets):
        return self.logits + offsets


def distributed_examples():
    offsets = torch.tensor([-.4, .1, .5, -.2]).reshape(4, 1, 1, 1)
    targets = torch.tensor([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1],
    ]).reshape(4, 1, 2, 3).float()
    return offsets, targets


def distributed_loss_worker(rank, rendezvous_uri, result_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        backend='gloo', init_method=rendezvous_uri, world_size=2, rank=rank,
        timeout=timedelta(seconds=45),
    )
    try:
        model = DistributedDataParallel(SharedLogits())
        offsets, targets = distributed_examples()
        indices = slice(rank * 2, (rank + 1) * 2)
        result = segmentation_loss(model(offsets[indices]), targets[indices], sync_dice=True)
        result['loss'].backward()
        values = torch.stack([result[key].detach() for key in
                              ('soft_dice', 'loss', 'log_soft_pq_loss', 'bce_loss')])
        gathered_values = [torch.empty_like(values) for _ in range(2)]
        dist.all_gather(gathered_values, values)
        gradients = [torch.empty_like(model.module.logits.grad) for _ in range(2)]
        dist.all_gather(gradients, model.module.logits.grad)
        if rank == 0:
            Path(result_path).write_text(json.dumps({
                'values': [item.tolist() for item in gathered_values],
                'gradients': [item.flatten().tolist() for item in gradients],
            }), encoding='utf-8')
    finally:
        dist.destroy_process_group()


def validation_totals(logits, targets, batches):
    """Accumulate the runtime's public metric inputs for a chosen partition."""
    totals = {name: 0.0 for name in (
        'loss', 'log_soft_pq_loss', 'bce_loss', 'soft_dice',
        'soft_intersection', 'predicted_area', 'target_area',
        'hard_intersection', 'hard_denominator', 'samples',
    )}
    with torch.no_grad():
        for indices in batches:
            if not indices:
                continue
            local_logits, local_targets = logits[indices], targets[indices]
            result = segmentation_loss(local_logits, local_targets)
            for name in ('loss', 'log_soft_pq_loss', 'bce_loss', 'soft_dice'):
                totals[name] += result[name].item() * len(indices)
            for name, value in zip(('soft_intersection', 'predicted_area', 'target_area'), result['overlap']):
                totals[name] += value.item()
            prediction, truth = local_logits.ge(0), local_targets.bool()
            totals['hard_intersection'] += (prediction & truth).sum().item()
            totals['hard_denominator'] += (prediction.sum() + truth.sum()).item()
            totals['samples'] += len(indices)
    return totals


class SegmentationLossTests(unittest.TestCase):
    def test_default_components_and_original_pq_reference(self):
        logits = torch.zeros(2, 1, 2, 2, requires_grad=True)
        targets = torch.tensor([[[[1., 0.], [0., 0.]]], [[[0., 0.], [0., 0.]]]])
        result = segmentation_loss(logits, targets)
        self.assertEqual(set(result), {'loss', 'log_soft_pq_loss', 'soft_dice', 'bce_loss', 'overlap'})
        # One positive tile: SQ=1.5/3.5, RQ=2/4. One empty tile: loss=.5.
        original_pq = (-.5 * math.log((1.5 / 3.5) * (2. / 4.)) + .5) / 2.
        global_dice = 2. / 6.
        expected_loss = .4 * original_pq + .4 * (1. - global_dice) + .2 * math.log(2.)
        self.assertAlmostEqual(result['log_soft_pq_loss'].item(), original_pq, places=6)
        self.assertAlmostEqual(result['soft_dice'].item(), global_dice, places=6)
        self.assertAlmostEqual(result['bce_loss'].item(), math.log(2.), places=6)
        self.assertAlmostEqual(result['loss'].item(), expected_loss, places=6)
        torch.testing.assert_close(result['overlap'], torch.tensor([.5, 4., 1.]))
        torch.testing.assert_close(log_soft_panoptic_quality_loss(logits, targets),
                                   result['log_soft_pq_loss'])
        # The compatibility helper must also preserve caller-selected smoothing.
        smooth = .25
        old_custom = (-.5 * math.log((.75 / 2.75) * (1.25 / 3.25)) + .5) / 2.
        self.assertAlmostEqual(log_soft_panoptic_quality_loss(logits, targets, smooth).item(),
                               old_custom, places=6)

    def test_confident_errors_and_single_pixel_masks_have_useful_finite_gradients(self):
        for height, width, foreground in ((2, 3, 'full'), (2, 3, 'empty'), (1, 1, 'full'), (1, 1, 'empty')):
            with self.subTest(shape=(height, width), foreground=foreground):
                positive = foreground == 'full'
                targets = torch.full((1, 1, height, width), float(positive))
                logits = torch.full_like(targets, -100. if positive else 100., requires_grad=True)
                result = segmentation_loss(logits, targets)
                for name, value in result.items():
                    self.assertTrue(torch.isfinite(value).all().item(), name)
                result['loss'].backward()
                self.assertTrue(torch.isfinite(logits.grad).all().item())
                signed_gradient = -logits.grad if positive else logits.grad
                self.assertTrue((signed_gradient > 0).all().item())
                # BCE must retain a useful correction even when sigmoid saturates.
                self.assertGreater(signed_gradient.min().item(), .1 / targets.numel())

    def test_sparse_foreground_and_background_gradient_directions(self):
        targets = torch.zeros(1, 1, 3, 3)
        targets[0, 0, 1, 1] = 1.
        logits = torch.zeros_like(targets, requires_grad=True)
        segmentation_loss(logits, targets)['loss'].backward()
        self.assertLess(logits.grad[targets.bool()].item(), 0.)
        self.assertTrue((logits.grad[~targets.bool()] > 0).all().item())

    def test_better_predictions_improve_pq_dice_bce_and_composite(self):
        targets = torch.tensor([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]).reshape(3, 1, 2, 3).float()
        wrong_logits = (1. - 2. * targets) * 2.
        better_logits = (2. * targets - 1.) * 2.
        before, after = [segmentation_loss(x, targets) for x in (wrong_logits, better_logits)]
        for name in ('loss', 'log_soft_pq_loss', 'bce_loss'):
            self.assertLess(after[name].item(), before[name].item(), name)
        self.assertGreater(after['soft_dice'].item(), before['soft_dice'].item())

    def test_validation_dice_and_selection_are_invariant_to_batching_and_rank_partition(self):
        logits = torch.tensor([
            [4., 3., -2., -1.],
            [-3., -2., -4., -1.],
            [0., -.2, .4, 1.],
            [2., 2., 2., 2.],
            [-1., 3., -2., .5],
        ]).reshape(5, 1, 2, 2)
        targets = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
        ]).reshape_as(logits).float()
        full = validation_totals(logits, targets, [[0, 1, 2, 3, 4]])
        batched = validation_totals(logits, targets, [[0, 1], [2, 3], [4]])
        # Uneven rank lengths and an empty rank must reduce without padded tiles.
        rank_totals = [validation_totals(logits, targets, batches) for batches in
                       ([[0, 3]], [[1, 2], [4]], [])]
        reduced = {key: sum(rank[key] for rank in rank_totals) for key in full}
        metrics = [summarize_epoch_totals(totals, DEFAULT_LOSS_WEIGHTS, training=False)
                   for totals in (full, batched, reduced)]
        for actual in metrics[1:]:
            for name, expected in metrics[0].items():
                self.assertAlmostEqual(actual[name], expected, places=6, msg=name)
        probability = logits.double().sigmoid()
        intersection = (probability * targets).sum().item()
        expected_soft = (2. * intersection + 1.) / (probability.sum().item() + targets.sum().item() + 1.)
        predicted = logits.ge(0)
        expected_hard = (2. * (predicted & targets.bool()).sum().item() + 1.) / (
            predicted.sum().item() + targets.sum().item() + 1.)
        self.assertAlmostEqual(metrics[0]['soft_dice'], expected_soft, places=6)
        self.assertAlmostEqual(metrics[0]['dice_at_0_5'], expected_hard, places=6)
        expected_selection = (
            .4 * metrics[0]['log_soft_pq_loss'] + .4 * (1. - expected_hard)
            + .2 * metrics[0]['bce_loss']
        )
        self.assertAlmostEqual(metrics[0]['selection_loss'], expected_selection, places=6)
        # This example distinguishes whole-set Dice from mean tile Dice.
        tile_average = sum(segmentation_loss(logits[i:i + 1], targets[i:i + 1])['soft_dice'].item()
                           for i in range(len(logits))) / len(logits)
        self.assertGreater(abs(expected_soft - tile_average), .001)

    def test_logged_pq_gain_does_not_outweigh_larger_dice_drop_in_selection(self):
        metrics = []
        # Integer counts yield the logged Dice values to four decimal places.
        for pq_loss, hard_intersection in ((.3203, 320000), (.3189, 311650)):
            totals = dict(samples=1, log_soft_pq_loss=pq_loss, bce_loss=.2,
                          soft_intersection=320000., predicted_area=500000., target_area=500000.,
                          hard_intersection=hard_intersection, hard_denominator=999999)
            metrics.append(summarize_epoch_totals(totals, DEFAULT_LOSS_WEIGHTS, training=False))
        self.assertEqual([round(x['dice_at_0_5'], 4) for x in metrics], [.6400, .6233])
        self.assertLess(metrics[1]['log_soft_pq_loss'], metrics[0]['log_soft_pq_loss'])
        self.assertGreater(metrics[1]['selection_loss'], metrics[0]['selection_loss'])
        self.assertAlmostEqual(metrics[1]['selection_loss'] - metrics[0]['selection_loss'],
                               .00612, places=6)

    def test_run_epoch_training_and_validation_use_composite_metrics(self):
        offsets, targets = distributed_examples()
        model = SharedLogits()
        initial = model.logits.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        config = {'amp': False, 'profile': False, 'log_every': 100000,
                  'loss_weights': DEFAULT_LOSS_WEIGHTS}
        train_batches = [(offsets[:2], targets[:2]), (offsets[2:], targets[2:])]
        validation_batches = [(offsets[:3], targets[:3]), (offsets[3:], targets[3:])]
        # Exercise the epoch itself on CPU while replacing only CUDA transfer/timing.
        with patch.object(runtime, 'CUDAPrefetcher', side_effect=lambda loader, *args: iter(loader)), \
                patch.object(torch.cuda, 'synchronize'):
            train, elapsed, rate = runtime.run_epoch(
                model, model, train_batches, optimizer, scaler, config, torch.device('cpu'),
                distributed=False, rank=0, world_size=1, training=True, epoch=2, channels_last=False,
            )
            self.assertTrue(math.isfinite(train['loss']))
            self.assertTrue(0. < train['soft_dice'] < 1.)
            self.assertTrue(math.isnan(train['dice_at_0_5']))
            self.assertGreater(elapsed, 0.)
            self.assertGreater(rate, 0.)
            self.assertFalse(torch.equal(initial, model.logits.detach()))
            actual, elapsed, rate = runtime.run_epoch(
                model, model, validation_batches, None, None, config, torch.device('cpu'),
                distributed=False, rank=0, world_size=1, training=False, epoch=2, channels_last=False,
            )
        with torch.no_grad():
            totals = validation_totals(model(offsets), targets, [[0, 1, 2], [3]])
        expected = summarize_epoch_totals(totals, DEFAULT_LOSS_WEIGHTS, training=False)
        for name in expected:
            self.assertAlmostEqual(actual[name], expected[name], places=6, msg=name)
        self.assertGreater(elapsed, 0.)
        self.assertGreater(rate, 0.)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU/Gloo is unavailable')
    def test_real_ddp_matches_concatenated_loss_and_parameter_gradients(self):
        offsets, targets = distributed_examples()
        model = SharedLogits()
        reference = segmentation_loss(model(offsets), targets)
        reference['loss'].backward()
        with tempfile.TemporaryDirectory(prefix='solar-loss-ddp-') as directory:
            result_path = Path(directory) / 'results.json'
            rendezvous_uri = (Path(directory) / 'rendezvous').as_uri()
            context = mp.spawn(distributed_loss_worker, args=(rendezvous_uri, str(result_path)),
                               nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 90.
                while not context.join(timeout=1.):
                    if time.monotonic() > deadline:
                        self.fail('Two-rank Gloo regression timed out')
                result = json.loads(result_path.read_text(encoding='utf-8'))
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5.)
        values = torch.tensor(result['values'])
        for rank in range(2):
            self.assertAlmostEqual(values[rank, 0].item(), reference['soft_dice'].item(), places=6)
            torch.testing.assert_close(torch.tensor(result['gradients'][rank]), model.logits.grad.flatten(),
                                       rtol=2e-5, atol=2e-7)
        averaged = values.mean(dim=0)
        for column, name in enumerate(('soft_dice', 'loss', 'log_soft_pq_loss', 'bce_loss')):
            self.assertAlmostEqual(averaged[column].item(), reference[name].item(), places=6, msg=name)


if __name__ == '__main__':
    unittest.main()
