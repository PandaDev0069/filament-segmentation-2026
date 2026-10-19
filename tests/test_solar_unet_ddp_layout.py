"""Exact gradient-layout regression for checkpointed channels-last DDP training."""

import contextlib
from datetime import timedelta
import io
from pathlib import Path
import tempfile
import time
import unittest
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

import solar_unet_full_image_runtime as runtime


def _small_unet():
    torch.manual_seed(629)
    return runtime.FullImageUNet(
        base_channels=8, depth=2, max_channels=16, checkpoint_blocks=True,
    )


def _rank_examples(rank):
    images = ((torch.arange(5 * 16 * 16).reshape(5, 1, 16, 16) * 7
               + rank * 43) % 256).to(torch.uint8)
    targets = torch.zeros_like(images)
    for index in range(5):
        targets[index, :, 3 + index, 1:15] = 1
        targets[index, :, 1:15, 5 + rank] = 1
    return images, targets


class _LayoutCheckingSGD(torch.optim.SGD):
    def __init__(self, module, **kwargs):
        super().__init__(module.parameters(), **kwargs)
        self.module = module
        self.layout_mismatches = []
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        for name, parameter in self.module.named_parameters():
            if parameter.grad is None or parameter.grad.stride() != parameter.stride():
                self.layout_mismatches.append((
                    self.steps, name, tuple(parameter.stride()),
                    None if parameter.grad is None else tuple(parameter.grad.stride()),
                ))
        return super().step(closure)


def _ddp_layout_worker(rank, rendezvous_uri, result_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous_uri, world_size=2, rank=rank,
                            timeout=timedelta(seconds=45))
    try:
        module = _small_unet().to(memory_format=torch.channels_last)
        projection = module.bottleneck[1].global_projection.weight
        incoming_strides = []

        def canonicalize_singleton_gradient(gradient):
            # CUDA can return [C, 1, 1, 1] for this 1x1 kernel, whereas DDP's
            # channels-last bucket expects [C, 1, C, C]. CPU kernels do not
            # reliably reproduce that choice. Materialize the identical values
            # with the observed CUDA strides so this is a deterministic test of
            # the production fix rather than a backend-dependent non-reproducer.
            canonical = torch.empty_strided(
                gradient.size(), (gradient.size(1), 1, 1, 1),
                dtype=gradient.dtype, device=gradient.device,
            ).copy_(gradient)
            incoming_strides.append(tuple(canonical.stride()))
            return canonical

        hook = projection.register_hook(canonicalize_singleton_gradient)
        model = DistributedDataParallel(module)
        optimizer = _LayoutCheckingSGD(module, lr=.03)
        images, targets = _rank_examples(rank)
        loader = DataLoader(TensorDataset(images, targets), batch_size=1, shuffle=False)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        config = {'amp': False, 'log_every': 100000,
                  'loss_weights': runtime.DEFAULT_LOSS_WEIGHTS,
                  'gradient_accumulation_steps': 2}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with contextlib.redirect_stdout(io.StringIO()):
                runtime.run_epoch(
                    model, module, loader, optimizer, scaler, config, torch.device('cpu'),
                    distributed=True, rank=rank, world_size=2, training=True, epoch=1,
                    channels_last=True,
                )
        hook.remove()
        local_result = {
            'steps': optimizer.steps,
            'layout_mismatches': optimizer.layout_mismatches,
            'warnings': [str(item.message) for item in captured],
            'incoming_strides': incoming_strides,
            'projection_stride': tuple(projection.stride()),
            'final_projection_grad_stride': (None if projection.grad is None
                                             else tuple(projection.grad.stride())),
        }
        rank_results = [None, None]
        dist.all_gather_object(rank_results, local_result)
        flattened = torch.cat([parameter.detach().flatten() for parameter in module.parameters()])
        parameters = [torch.empty_like(flattened) for _ in range(2)]
        dist.all_gather(parameters, flattened)
        if rank == 0:
            torch.save({'ranks': rank_results, 'parameters': parameters}, result_path)
    finally:
        dist.destroy_process_group()


class GradientLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_gradient_buffers_preserve_exact_singleton_strides_and_existing_buffers(self):
        module = nn.Sequential(nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1))
        module.to(memory_format=torch.channels_last)
        missing = module[0].weight
        existing = module[1].weight
        existing.grad = torch.empty_strided(existing.size(), existing.stride()).fill_(7.)
        existing_buffer = existing.grad
        existing_pointer = existing.grad.data_ptr()
        module[1].bias.requires_grad_(False)

        runtime.initialize_gradient_buffers(module)

        self.assertEqual(missing.stride(), (4, 1, 4, 4))
        self.assertEqual(missing.grad.stride(), missing.stride())
        self.assertEqual(missing.grad.dtype, missing.dtype)
        self.assertEqual(missing.grad.device, missing.device)
        self.assertEqual(torch.count_nonzero(missing.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(module[0].bias.grad).item(), 0)
        self.assertIs(existing.grad, existing_buffer)
        self.assertEqual(existing.grad.data_ptr(), existing_pointer)
        torch.testing.assert_close(existing.grad, torch.full_like(existing.grad, 7.))
        self.assertIsNone(module[1].bias.grad)

        runtime.initialize_gradient_buffers(module)
        self.assertIs(existing.grad, existing_buffer)
        torch.testing.assert_close(existing.grad, torch.full_like(existing.grad, 7.))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU/Gloo is unavailable')
    def test_two_rank_checkpointed_channels_last_accumulation_keeps_exact_grad_layout(self):
        # One globally pooled Dice objective per microbatch, averaged over
        # windows of 2, 2, and 1; this is the same objective used by run_epoch.
        reference = _small_unet()
        optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        rank_examples = [_rank_examples(rank) for rank in range(2)]
        for start in (0, 2, 4):
            end = min(start + 2, 5)
            optimizer.zero_grad(set_to_none=True)
            for index in range(start, end):
                images = torch.cat([data[0][index:index + 1] for data in rank_examples])
                targets = torch.cat([data[1][index:index + 1] for data in rank_examples])
                loss = runtime.segmentation_loss(reference(images.float() / 255.), targets)['loss']
                (loss / (end - start)).backward()
            optimizer.step()
        expected = torch.cat([parameter.detach().flatten() for parameter in reference.parameters()])

        with tempfile.TemporaryDirectory(prefix='solar-ddp-layout-') as directory:
            result_path = Path(directory) / 'result.pt'
            rendezvous_uri = (Path(directory) / 'rendezvous').as_uri()
            context = mp.spawn(_ddp_layout_worker, args=(rendezvous_uri, str(result_path)),
                               nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 90.
                while not context.join(timeout=1.):
                    if time.monotonic() > deadline:
                        self.fail('Two-rank gradient-layout regression timed out')
                result = torch.load(result_path, weights_only=True)
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5.)

        for rank, details in enumerate(result['ranks']):
            with self.subTest(rank=rank):
                self.assertEqual(details['steps'], 3)
                self.assertEqual(details['layout_mismatches'], [])
                self.assertEqual(details['projection_stride'], (16, 1, 16, 16))
                self.assertEqual(details['final_projection_grad_stride'], details['projection_stride'])
                self.assertEqual(details['incoming_strides'], [(16, 1, 1, 1)] * 5)
                self.assertFalse(any('Grad strides do not match bucket view strides' in message
                                     for message in details['warnings']), details['warnings'])
                torch.testing.assert_close(result['parameters'][rank], expected,
                                           rtol=2e-4, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
