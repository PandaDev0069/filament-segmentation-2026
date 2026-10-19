"""Regression checks for full-frame training and seam-free inference.

Run with ``python -m unittest discover -s tests -p test_solar_unet_full_image.py -v``.
The real U-Net is exercised at a small spatial scale; the 2048-square inference
contract is checked with a cheap model so these tests do not require a GPU.
"""

import contextlib
from datetime import timedelta
import io
import json
from pathlib import Path
import random
import tempfile
import time
import unittest

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

import solar_unet_full_image_runtime as runtime


class RecordingPixelModel(nn.Module):
    """Threshold the original full image, recording every model invocation."""

    depth = 7
    image_size = 2048

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, image):
        self.calls.append({
            'shape': tuple(image.shape),
            'dtype': image.dtype,
            'minimum': image.min().item(),
            'maximum': image.max().item(),
            'probes': image[0, 0, 1000, :4].tolist(),
        })
        return (image - .5) * 16.


class SharedPixelLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([-.6, .3, -.2, .8, -1., .4]).reshape(1, 1, 2, 3))

    def forward(self, image):
        return self.logits + image


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def distributed_full_image_examples(rank):
    images = (torch.arange(6 * 6).reshape(6, 1, 2, 3) * 3 + rank * 40).to(torch.uint8)
    targets = torch.zeros_like(images)
    for index in range(len(images)):
        targets[index, :, 0, (index + rank) % 3] = 1
        if (index + rank) % 2:
            targets[index, :, 1, :] = 1
    return images, targets


def full_image_ddp_accumulation_worker(rank, rendezvous_uri, result_path):
    torch.set_num_threads(1)
    dist.init_process_group(backend='gloo', init_method=rendezvous_uri, world_size=2, rank=rank,
                            timeout=timedelta(seconds=45))
    try:
        model = DistributedDataParallel(SharedPixelLogits())
        optimizer = CountingSGD(model.parameters(), lr=.15)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        images, targets = distributed_full_image_examples(rank)
        loader = DataLoader(TensorDataset(images, targets), batch_size=1, shuffle=False)
        config = {'amp': False, 'log_every': 100000,
                  'loss_weights': runtime.DEFAULT_LOSS_WEIGHTS, 'gradient_accumulation_steps': 4}
        runtime.run_epoch(model, model.module, loader, optimizer, scaler, config,
                          torch.device('cpu'), distributed=True, rank=rank, world_size=2,
                          training=True, epoch=0, channels_last=False)
        parameters = [torch.empty_like(model.module.logits) for _ in range(2)]
        dist.all_gather(parameters, model.module.logits.detach())
        if rank == 0:
            Path(result_path).write_text(json.dumps({
                'parameters': [value.flatten().tolist() for value in parameters],
                'steps': optimizer.steps,
            }), encoding='utf-8')
    finally:
        dist.destroy_process_group()


class FullImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_seven_downsamplings_support_batch_one_and_finite_training_gradients(self):
        torch.manual_seed(33)
        model = runtime.FullImageUNet(
            base_channels=16, depth=7, max_channels=32, checkpoint_blocks=True,
        ).train()
        self.assertTrue(any(isinstance(layer, nn.GroupNorm) for layer in model.modules()))
        self.assertFalse(any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
                             for layer in model.modules()))
        spatial_shapes = []
        hooks = [layer.register_forward_hook(
            lambda module, arguments, output: spatial_shapes.append(tuple(output.shape[-2:])))
            for layer in model.modules() if isinstance(layer, nn.GroupNorm)]
        try:
            image = torch.rand(1, 1, 256, 256)
            targets = torch.zeros_like(image)
            targets[:, :, 120:123, 10:240] = 1.
            logits = model(image)
            self.assertEqual(tuple(logits.shape), tuple(image.shape))
            self.assertEqual(min(shape[0] for shape in spatial_shapes), 2)
            self.assertTrue(torch.isfinite(logits).all().item())
            runtime.segmentation_loss(logits, targets)['loss'].backward()
        finally:
            for hook in hooks:
                hook.remove()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)

    def test_checkpointing_preserves_outputs_and_all_parameter_gradients(self):
        torch.manual_seed(41)
        plain = runtime.FullImageUNet(
            base_channels=16, depth=7, max_channels=32, checkpoint_blocks=False,
        ).train()
        checkpointed = runtime.FullImageUNet(
            base_channels=16, depth=7, max_channels=32, checkpoint_blocks=True,
        ).train()
        checkpointed.load_state_dict(plain.state_dict())
        image = torch.rand(1, 1, 256, 256)
        targets = image.gt(.7).float()
        outputs = []
        for model in (plain, checkpointed):
            output = model(image)
            runtime.segmentation_loss(output, targets)['loss'].backward()
            outputs.append(output.detach())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-6, atol=1e-7)
        plain_parameters = dict(plain.named_parameters())
        checkpointed_parameters = dict(checkpointed.named_parameters())
        self.assertEqual(set(plain_parameters), set(checkpointed_parameters))
        for name, expected in plain_parameters.items():
            actual = checkpointed_parameters[name]
            self.assertIsNotNone(expected.grad, name)
            self.assertIsNotNone(actual.grad, name)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7, msg=name)

    def test_native_prediction_uses_one_forward_and_preserves_cross_boundary_filament(self):
        image = np.zeros((2048, 2048), dtype=np.uint8)
        image[1000, :] = 255
        image[1000, :4] = (128, 192, 224, 255)
        image[200:1800, 1024] = 255
        model = RecordingPixelModel()
        probabilities = runtime.predict_full_image(model, image, torch.device('cpu'))
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertEqual(call['shape'], (1, 1, 2048, 2048))
        self.assertEqual(call['dtype'], torch.float32)
        self.assertEqual((call['minimum'], call['maximum']), (0., 1.))
        np.testing.assert_allclose(call['probes'], image[1000, :4].astype(np.float32) / 255.)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape, image.shape)
        self.assertEqual(probabilities.dtype, np.float32)
        self.assertTrue(np.isfinite(probabilities).all())
        predicted = probabilities >= .5
        np.testing.assert_array_equal(predicted, image >= 128)
        for boundary in (512, 1024, 1536):
            self.assertTrue(predicted[1000, boundary - 1:boundary + 1].all())
        self.assertEqual(cv2.connectedComponents(predicted.astype(np.uint8), connectivity=8)[0], 2)

    def test_native_predictor_rejects_tiles_before_calling_model(self):
        model = RecordingPixelModel()
        with self.assertRaises(ValueError):
            runtime.predict_full_image(model, np.zeros((512, 512), dtype=np.uint8),
                                       torch.device('cpu'))
        self.assertEqual(model.calls, [])

    def _write_example(self, directory, image, mask):
        image_path = Path(directory) / 'image.png'
        mask_path = Path(directory) / 'mask.png'
        self.assertTrue(cv2.imwrite(str(image_path), image))
        self.assertTrue(cv2.imwrite(str(mask_path), mask))
        return [{'image_path': str(image_path), 'mask_path': str(mask_path),
                 'image_id': 1, 'file_name': image_path.name}]

    def test_dataset_retains_native_pixels_and_binarizes_mask(self):
        image = np.tile(np.arange(2048, dtype=np.uint16) % 256, (2048, 1)).astype(np.uint8)
        mask = np.zeros_like(image)
        mask[100:150, 200:1700] = 255
        with tempfile.TemporaryDirectory(prefix='solar-full-image-') as directory:
            records = self._write_example(directory, image, mask)
            dataset = runtime.FullImageDataset(records)
            actual_image, actual_mask = dataset[0]
        self.assertEqual(len(dataset), 1)
        self.assertEqual(tuple(actual_image.shape), (1, 2048, 2048))
        self.assertEqual(tuple(actual_mask.shape), (1, 2048, 2048))
        self.assertEqual(actual_image.dtype, torch.uint8)
        self.assertEqual(actual_mask.dtype, torch.uint8)
        np.testing.assert_array_equal(actual_image[0].numpy(), image)
        np.testing.assert_array_equal(actual_mask[0].numpy(), mask > 0)

    def test_dataset_augmentation_keeps_image_and_mask_aligned(self):
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        mask[100:130, 300:1800] = 255
        mask[100:1900, 1200:1220] = 255
        transformed = []
        with tempfile.TemporaryDirectory(prefix='solar-full-augment-') as directory:
            records = self._write_example(directory, mask, mask)
            dataset = runtime.FullImageDataset(records, augment=True)
            random_state = random.getstate()
            try:
                for seed in range(4):
                    random.seed(seed)
                    image, target = dataset[0]
                    self.assertEqual(tuple(image.shape), (1, 2048, 2048))
                    torch.testing.assert_close(image.gt(0), target.bool())
                    transformed.append(not np.array_equal(image[0].numpy(), mask))
            finally:
                random.setstate(random_state)
        self.assertTrue(any(transformed), 'At least one deterministic seed should transform the frame')

    def test_dataset_rejects_image_or_mask_with_wrong_native_shape(self):
        for image_shape, mask_shape in (((1024, 1024), (1024, 1024)),
                                        ((2048, 2048), (2048, 2047))):
            with self.subTest(image=image_shape, mask=mask_shape):
                with tempfile.TemporaryDirectory(prefix='solar-full-invalid-') as directory:
                    records = self._write_example(directory, np.zeros(image_shape, dtype=np.uint8),
                                                  np.zeros(mask_shape, dtype=np.uint8))
                    with self.assertRaises(ValueError):
                        runtime.FullImageDataset(records)[0]

    def test_accumulation_matches_weighted_microbatches_including_partial_window(self):
        images = (torch.arange(11 * 6).reshape(11, 1, 2, 3) * 3).to(torch.uint8)
        targets = torch.zeros_like(images)
        for index in range(len(images)):
            targets[index, :, 0, index % 3] = 1
            if index % 2:
                targets[index, :, 1, :] = 1
        loader = DataLoader(TensorDataset(images, targets), batch_size=2, shuffle=False)
        batches = list(loader)
        self.assertEqual([image.shape[0] for image, target in batches], [2, 2, 2, 2, 2, 1])
        reference = SharedPixelLogits()
        actual = SharedPixelLogits()
        actual.load_state_dict(reference.state_dict())
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.15)
        for start in (0, 4):
            window = batches[start:start + 4]
            reference_optimizer.zero_grad(set_to_none=True)
            total_images = sum(image.shape[0] for image, target in window)
            for image, target in window:
                result = runtime.segmentation_loss(reference(image.float() / 255.), target)
                (result['loss'] * image.shape[0] / total_images).backward()
            reference_optimizer.step()
        optimizer = CountingSGD(actual.parameters(), lr=.15)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        config = {'amp': False, 'profile': False, 'log_every': 100000,
                  'loss_weights': runtime.DEFAULT_LOSS_WEIGHTS, 'gradient_accumulation_steps': 4}
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, seconds, rate = runtime.run_epoch(
                actual, actual, loader, optimizer, scaler, config, torch.device('cpu'),
                distributed=False, rank=0, world_size=1, training=True, epoch=0,
                channels_last=False,
            )
        self.assertEqual(optimizer.steps, 2)
        torch.testing.assert_close(actual.logits, reference.logits, rtol=1e-6, atol=1e-7)
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertGreater(seconds, 0.)
        self.assertGreater(rate, 0.)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU/Gloo is unavailable')
    def test_real_two_rank_accumulation_matches_global_microbatch_reference(self):
        rank_examples = [distributed_full_image_examples(rank) for rank in range(2)]
        reference = SharedPixelLogits()
        optimizer = torch.optim.SGD(reference.parameters(), lr=.15)
        for start in (0, 4):
            end = min(start + 4, 6)
            optimizer.zero_grad(set_to_none=True)
            for index in range(start, end):
                images = torch.cat([data[0][index:index + 1] for data in rank_examples])
                targets = torch.cat([data[1][index:index + 1] for data in rank_examples])
                loss = runtime.segmentation_loss(reference(images.float() / 255.), targets)['loss']
                (loss / (end - start)).backward()
            optimizer.step()
        with tempfile.TemporaryDirectory(prefix='solar-full-ddp-') as directory:
            result_path = Path(directory) / 'result.json'
            rendezvous_uri = (Path(directory) / 'rendezvous').as_uri()
            context = mp.spawn(full_image_ddp_accumulation_worker,
                               args=(rendezvous_uri, str(result_path)), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 90.
                while not context.join(timeout=1.):
                    if time.monotonic() > deadline:
                        self.fail('Two-rank accumulation regression timed out')
                result = json.loads(result_path.read_text(encoding='utf-8'))
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5.)
        self.assertEqual(result['steps'], 2)
        for actual in result['parameters']:
            torch.testing.assert_close(torch.tensor(actual), reference.logits.detach().flatten(),
                                       rtol=2e-5, atol=2e-7)


if __name__ == '__main__':
    unittest.main()
