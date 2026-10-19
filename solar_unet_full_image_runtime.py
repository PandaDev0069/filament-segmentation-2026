"""Native-resolution U-Net worker; launched with torchrun from the full-image notebook.

No tiles or spatial resampling are used. DDP replicates the model on each GPU;
it does not combine the cards into one larger VRAM pool.
"""
import argparse
from contextlib import nullcontext
import csv
import gc
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        groups = math.gcd(8, out_channels)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=False),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class GlobalContext(nn.Module):
    """A pooled image summary makes every output depend on the complete input."""
    def __init__(self, channels):
        super().__init__()
        self.local = DoubleConv(channels, channels, dilation=2)
        self.global_projection = nn.Conv2d(channels, channels, 1)

    def forward(self, inputs):
        local = self.local(inputs)
        return local + self.global_projection(F.adaptive_avg_pool2d(local, 1))


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, 2, stride=2)
        self.conv = DoubleConv(skip_channels * 2, skip_channels)

    def forward(self, inputs, skip):
        return self.conv(torch.cat((self.up(inputs), skip), dim=1))


class FullImageUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=16,
                 depth=7, max_channels=256, checkpoint_blocks=True):
        super().__init__()
        if min(in_channels, out_channels, base_channels, depth, max_channels) < 1:
            raise ValueError('Channel counts and depth must be positive integers.')
        self.depth = int(depth)
        self.image_size = 2048
        self.checkpoint_blocks = bool(checkpoint_blocks)
        channels = [min(base_channels * 2**index, max_channels)
                    for index in range(self.depth + 1)]
        self.encoders = nn.ModuleList()
        current = in_channels
        for channel in channels[:-1]:
            self.encoders.append(DoubleConv(current, channel))
            current = channel
        self.bottleneck = nn.Sequential(DoubleConv(current, channels[-1]),
                                        GlobalContext(channels[-1]))
        self.decoders = nn.ModuleList()
        current = channels[-1]
        for channel in reversed(channels[:-1]):
            self.decoders.append(DecoderStage(current, channel))
            current = channel
        self.output = nn.Conv2d(current, out_channels, 1)

    def _block(self, block, *inputs):
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():
            # Checkpointing the complete decoder stage also releases up/cat tensors.
            return checkpoint(block, *inputs, use_reentrant=False)
        return block(*inputs)

    def forward(self, inputs):
        divisor = 2**self.depth
        if inputs.ndim != 4 or min(inputs.shape[-2:]) < divisor or any(
                size % divisor for size in inputs.shape[-2:]):
            raise ValueError(f'Input must be BCHW, with H and W divisible by {divisor}.')
        skips = []
        for encoder in self.encoders:
            inputs = self._block(encoder, inputs)
            skips.append(inputs)
            inputs = F.max_pool2d(inputs, 2)
        inputs = self._block(self.bottleneck, inputs)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            inputs = self._block(decoder, inputs, skip)
        return self.output(inputs)


DEFAULT_LOSS_WEIGHTS = {'log_soft_pq': 0.4, 'dice': 0.4, 'bce': 0.2}


def validate_loss_weights(weights):
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError(f'loss_weights must contain exactly {tuple(DEFAULT_LOSS_WEIGHTS)}')
    weights = {name: float(value) for name, value in weights.items()}
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError('Loss weights must be finite and nonnegative.')
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError('Loss weights must sum to 1.')
    return weights


def segmentation_loss(logits, targets, weights=None, sync_dice=False, smooth=1.0):
    """Original union-PQ surrogate, global foreground Dice and pixel BCE.

    This is a semantic union-mask surrogate, not true instance panoptic quality.
    sync_dice pools each microbatch across ranks. Accumulation averages these
    microbatch Dice objectives; it is not Dice over the entire effective batch.
    """
    weights = DEFAULT_LOSS_WEIGHTS if weights is None else weights
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        probabilities = logits.sigmoid()
        targets = targets.float()
        dims = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dims)
        predicted_area = probabilities.sum(dim=dims)
        target_area = targets.sum(dim=dims)
        total_area = predicted_area + target_area
        soft_sq = (intersection + smooth) / (total_area - intersection + smooth)
        soft_rq = (2.0 * intersection + smooth) / (total_area + smooth)
        positive_loss = -0.5 * (soft_sq.clamp_min(1e-6).log()
                                + soft_rq.clamp_min(1e-6).log())
        empty_loss = predicted_area / float(targets.shape[-2] * targets.shape[-1])
        pq_loss = torch.where(target_area > 0, positive_loss, empty_loss).mean()
        overlap = torch.stack((intersection.sum(), predicted_area.sum(), target_area.sum()))
        dice_overlap = overlap
        if sync_dice:
            global_overlap = overlap.detach().clone()
            dist.all_reduce(global_overlap)
            dice_overlap = global_overlap + dist.get_world_size() * (overlap - overlap.detach())
        soft_dice = (2.0 * dice_overlap[0] + smooth) / (
            dice_overlap[1] + dice_overlap[2] + smooth)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = (weights['log_soft_pq'] * pq_loss
                + weights['dice'] * (1.0 - soft_dice) + weights['bce'] * bce_loss)
        return dict(loss=loss, log_soft_pq_loss=pq_loss, soft_dice=soft_dice,
                    bce_loss=bce_loss, overlap=overlap)


def log_soft_panoptic_quality_loss(logits, targets, smooth=1.0):
    return segmentation_loss(logits, targets, smooth=smooth)['log_soft_pq_loss']


def summarize_epoch_totals(totals, weights, training):
    samples = totals['samples']
    if samples == 0:
        raise ValueError('Empty epoch. Check the image records and batch size.')
    pq_loss = totals['log_soft_pq_loss'] / samples
    bce_loss = totals['bce_loss'] / samples
    if training:
        soft_dice = totals['soft_dice'] / samples
        hard_dice = selection_loss = float('nan')
        loss = totals['loss'] / samples
    else:
        soft_dice = (2.0 * totals['soft_intersection'] + 1.0) / (
            totals['predicted_area'] + totals['target_area'] + 1.0)
        hard_dice = (2.0 * totals['hard_intersection'] + 1.0) / (
            totals['hard_denominator'] + 1.0)
        loss = (weights['log_soft_pq'] * pq_loss + weights['dice'] * (1.0 - soft_dice)
                + weights['bce'] * bce_loss)
        selection_loss = (weights['log_soft_pq'] * pq_loss
                          + weights['dice'] * (1.0 - hard_dice) + weights['bce'] * bce_loss)
    return dict(loss=loss, log_soft_pq_loss=pq_loss, soft_dice=soft_dice,
                bce_loss=bce_loss, dice_at_0_5=hard_dice, selection_loss=selection_loss)


class FullImageDataset(Dataset):
    def __init__(self, records, augment=False, image_size=2048):
        self.records = list(records)
        self.augment = bool(augment)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # Decode JPEG directly as grayscale; never resize, crop or tile it.
        image = cv2.imread(str(record['image_path']), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(record['mask_path']), cv2.IMREAD_UNCHANGED)
        expected = (self.image_size, self.image_size)
        if image is None or mask is None:
            raise OSError(f'Unreadable full-image pair: {record["image_path"]}')
        if image.shape != expected or mask.shape != expected:
            raise ValueError(f'Native full-image pair must have shape {expected}; '
                             f'got {image.shape}, {mask.shape}. No resizing is allowed.')
        if image.dtype != np.uint8 or mask.dtype != np.uint8:
            raise ValueError('Full images and PNG masks must decode to uint8.')
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError('Full-image masks must contain only 0 and 255.')
        mask = (mask > 127).astype(np.uint8)
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if random.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)
            turns = random.randrange(4)
            if turns:
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        return (torch.from_numpy(np.array(image[None], dtype=np.uint8, order='C', copy=True)),
                torch.from_numpy(np.array(mask[None], dtype=np.uint8, order='C', copy=True)))


def seed_loader_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def accumulation_window_size(batch_index, total_batches, accumulation_steps):
    """Number of microbatches in this zero-based index's actual update window."""
    if accumulation_steps < 1 or not 0 <= batch_index < total_batches:
        raise ValueError('Invalid accumulation window arguments.')
    start = batch_index // accumulation_steps * accumulation_steps
    return min(accumulation_steps, total_batches - start)


def _gather_objects(value, distributed, world_size):
    if not distributed:
        return [value]
    values = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def _new_optimizer(module, config):
    # foreach=False avoids temporary tensor lists at the full-resolution VRAM peak.
    return torch.optim.Adam(module.parameters(), lr=config['learning_rate'],
                            weight_decay=config['weight_decay'], foreach=False)


def initialize_gradient_buffers(module):
    """Match exact parameter strides before DDP's first backward.

    Singleton dimensions in channels-last 1x1 kernels have ambiguous strides.
    Explicit buffers avoid recreating canonicalized gradients that do not match
    DDP's bucket views. Keep existing buffers, including established bucket views.
    """
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.requires_grad and parameter.grad is None:
                parameter.grad = torch.empty_strided(
                    parameter.size(), parameter.stride(),
                    dtype=parameter.dtype, device=parameter.device,
                ).zero_()


def _probe_candidate(module, batch_size, config, device):
    """Unwrapped, collective-free forward/backward/Adam probe, including slot allocation."""
    optimizer = scaler = images = targets = logits = components = None
    report = dict(batch_size=batch_size, fits=False, out_of_memory=False)
    gc.collect()
    torch.cuda.empty_cache()
    baseline_free, total = torch.cuda.mem_get_info(device)
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        optimizer = _new_optimizer(module, config)
        scaler = torch.amp.GradScaler('cuda', enabled=config['amp'], init_scale=1024.0)
        images = torch.rand((batch_size, 1, config['image_size'], config['image_size']),
                            device=device)
        targets = (torch.rand_like(images) > 0.98).float()
        if config['channels_last']:
            images = images.contiguous(memory_format=torch.channels_last)
        durations = []
        warmup = int(config.get('probe_warmup_steps', 2))
        measured = int(config.get('probe_measure_steps', 2))
        if warmup < 1 or measured < 1:
            raise ValueError('The CUDA probe requires at least one warmup and measured step.')
        for step in range(warmup + measured):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                logits = module(images)
            components = segmentation_loss(logits, targets, config['loss_weights'], sync_dice=False)
            scaler.scale(components['loss']).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            if step >= warmup:
                durations.append(time.perf_counter() - started)
            # Do not retain a prior graph during the next forward pass.
            logits = components = None
        if not optimizer.state:
            raise RuntimeError('Probe Adam state was not allocated; AMP skipped every step.')
        allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        free_after, _ = torch.cuda.mem_get_info(device)
        # Leave room for DDP gradient buckets, NCCL and allocator variation.
        parameter_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        ddp_margin = max(384 * 1024**2, parameter_bytes * 2)
        available_headroom = min(free_after,
                                 baseline_free - max(0, reserved - baseline_reserved))
        fraction_budget = total * config['memory_fraction']
        fits = allocated <= fraction_budget and reserved <= fraction_budget and available_headroom >= ddp_margin
        report.update(fits=bool(fits), peak_allocated_bytes=allocated,
                      peak_reserved_bytes=reserved, free_after_bytes=free_after,
                      total_bytes=total, available_headroom_bytes=available_headroom,
                      ddp_margin_bytes=ddp_margin, step_seconds=float(np.median(durations)),
                      images_per_second=batch_size / float(np.median(durations)),
                      optimizer_state_allocated=True)
    except torch.cuda.OutOfMemoryError:
        report.update(out_of_memory=True, error='CUDA out of memory in unwrapped probe',
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    finally:
        module.zero_grad(set_to_none=True)
        optimizer = scaler = images = targets = logits = components = None
        gc.collect()
        torch.cuda.empty_cache()
    return report


def probe_batch_sizes(module, config, device, distributed, rank, world_size):
    requested = config['per_gpu_batch_size']
    if requested == 'auto':
        candidates = sorted(set(int(value) for value in config.get('batch_candidates', [1, 2])))
    else:
        if int(requested) < 1:
            raise ValueError('per_gpu_batch_size must be positive or "auto".')
        candidates = sorted(set([1, int(requested)]))
    if not candidates or min(candidates) < 1:
        raise ValueError('batch_candidates must contain positive batch sizes.')
    saved_state = {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
    saved_rng = (torch.get_rng_state(), torch.cuda.get_rng_state(device),
                 random.getstate(), np.random.get_state())
    local_trials = []
    try:
        for batch_size in candidates:
            module.load_state_dict(saved_state)
            torch.set_rng_state(saved_rng[0])
            torch.cuda.set_rng_state(saved_rng[1], device)
            random.setstate(saved_rng[2])
            np.random.set_state(saved_rng[3])
            trial = _probe_candidate(module, batch_size, config, device)
            local_trials.append(trial)
            print(f'Rank {rank} full-image probe: batch={batch_size}; fits={trial["fits"]}; '
                  f'peak allocated={trial.get("peak_allocated_bytes", 0) / 1024**3:.2f} GiB; '
                  f'peak reserved={trial.get("peak_reserved_bytes", 0) / 1024**3:.2f} GiB', flush=True)
    finally:
        module.load_state_dict(saved_state)
        module.zero_grad(set_to_none=True)
        torch.set_rng_state(saved_rng[0])
        torch.cuda.set_rng_state(saved_rng[1], device)
        random.setstate(saved_rng[2])
        np.random.set_state(saved_rng[3])
        del saved_state
        gc.collect()
        torch.cuda.empty_cache()
    # All collectives happen after local OOM handling and cleanup. Probes never use DDP.
    rank_trials = _gather_objects(dict(rank=rank, trials=local_trials), distributed, world_size)
    feasible = []
    for candidate in candidates:
        trials = [next(t for t in row['trials'] if t['batch_size'] == candidate) for row in rank_trials]
        if all(trial['fits'] for trial in trials):
            feasible.append((candidate * world_size / max(t['step_seconds'] for t in trials), candidate))
    if not feasible:
        raise RuntimeError('No full-image batch passed the real CUDA probe on every GPU. '
                           'Reduce base_channels/max_channels, keep AMP and checkpointing, '
                           'or free VRAM. This runtime will not tile or resize images.')
    selected = max(feasible)[1] if requested == 'auto' else (
        int(requested) if any(candidate == int(requested) for _, candidate in feasible) else 1)
    if rank == 0:
        print(f'Selected per-GPU full-image batch: {selected}; real allocation checks passed '
              'with a reserved margin. Training fit still depends on the actual CUDA runtime.', flush=True)
    return selected, rank_trials


def run_epoch(model, module, loader, optimizer, scaler, config, device,
              distributed, rank, world_size, training, epoch, channels_last):
    model.train(training)
    prediction_model = model if training else module
    total_names = ('loss', 'log_soft_pq_loss', 'bce_loss', 'soft_dice',
                   'soft_intersection', 'predicted_area', 'target_area',
                   'hard_intersection', 'hard_denominator', 'samples')
    totals = {name: torch.zeros((), dtype=torch.float64, device=device) for name in total_names}
    weights = config.get('loss_weights', DEFAULT_LOSS_WEIGHTS)
    accumulation = int(config.get('gradient_accumulation_steps', 4))
    if accumulation < 1:
        raise ValueError('gradient_accumulation_steps must be positive.')
    total_batches = len(loader)
    batch_size = loader.batch_size
    local_samples = len(loader.sampler)
    final_batch_size = local_samples - (total_batches - 1) * batch_size if total_batches else 0
    if loader.drop_last:
        final_batch_size = batch_size
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if training:
        if distributed:
            initialize_gradient_buffers(module)
        # Retain exact singleton strides and DDP bucket views between updates.
        optimizer.zero_grad(set_to_none=not distributed)
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index, (images, targets) in enumerate(loader):
            images = images.to(device, dtype=torch.float32, non_blocking=True).div_(255.0)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            window_count = accumulation_window_size(batch_index, total_batches, accumulation)
            window_start = batch_index // accumulation * accumulation
            window_end = window_start + window_count
            window_samples = window_count * batch_size
            if window_end == total_batches:
                window_samples -= batch_size - final_batch_size
            last_in_window = batch_index + 1 == window_end
            # The final short window (and any smaller final image batch) is weighted exactly.
            sync_context = model.no_sync() if training and distributed and not last_in_window else nullcontext()
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=config.get('amp', True) and device.type == 'cuda'):
                    logits = prediction_model(images)
                components = segmentation_loss(logits, targets, weights,
                                               sync_dice=training and distributed)
                if training:
                    scaler.scale(components['loss'] * (images.shape[0] / window_samples)).backward()
            if training and last_in_window:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=not distributed)
            for name in ('loss', 'log_soft_pq_loss', 'bce_loss', 'soft_dice'):
                totals[name] += components[name].detach().double() * images.shape[0]
            if not training:
                for name, value in zip(('soft_intersection', 'predicted_area', 'target_area'), components['overlap']):
                    totals[name] += value.detach().double()
                predictions, truth = logits.detach().ge(0), targets.bool()
                totals['hard_intersection'] += torch.logical_and(predictions, truth).sum()
                totals['hard_denominator'] += predictions.sum() + truth.sum()
            totals['samples'] += images.shape[0]
            if rank == 0 and (batch_index + 1) % config.get('log_every', 20) == 0:
                print(f'Epoch {epoch}: {"train" if training else "validation"} '
                      f'full-image batch {batch_index + 1}/{total_batches}', flush=True)
            # Release full-resolution loss intermediates before the following forward.
            del images, targets, logits, components
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    packed = torch.stack([totals[name] for name in total_names])
    slowest = torch.tensor(seconds, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(packed)
        dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
    totals = dict(zip(total_names, packed.cpu().tolist()))
    seconds = slowest.item()
    return summarize_epoch_totals(totals, weights, training), seconds, totals['samples'] / max(seconds, 1e-9)


def predict_full_image(model, image, device, amp=True, channels_last=True):
    """One complete native image -> one complete float32 probability map."""
    image = np.asarray(image)
    expected_size = int(getattr(model, 'image_size', 2048))
    if image.dtype != np.uint8 or image.shape != (expected_size, expected_size):
        raise ValueError(f'Prediction requires one native uint8 {expected_size}x{expected_size} grayscale image.')
    divisor = 2**int(getattr(model, 'depth', 7))
    if expected_size % divisor:
        raise ValueError(f'Native image dimensions must be divisible by {divisor}.')
    device = torch.device(device)
    inputs = torch.from_numpy(np.array(image[None, None], order='C', copy=True)).to(
        device, dtype=torch.float32).div_(255.0)
    if channels_last:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    model.eval()
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
        logits = model(inputs)
    if logits.shape != (1, 1, expected_size, expected_size):
        raise ValueError(f'Model returned an unexpected full-image shape: {tuple(logits.shape)}')
    return logits[0, 0].float().sigmoid().cpu().numpy()


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def train(config):
    config = dict(config)
    defaults = dict(image_size=2048, base_channels=16, depth=7, max_channels=256,
                    checkpoint_blocks=True, seed=42, epochs=30, patience=6,
                    learning_rate=1e-4, weight_decay=1e-5, amp=True, workers=4,
                    prefetch_factor=1, gradient_accumulation_steps=4,
                    per_gpu_batch_size=1, memory_fraction=0.85, channels_last=True,
                    cudnn_benchmark=False, log_every=20)
    for name, value in defaults.items():
        config.setdefault(name, value)
    config['loss_weights'] = validate_loss_weights(config.get('loss_weights', DEFAULT_LOSS_WEIGHTS))
    if not torch.cuda.is_available():
        raise RuntimeError('Full-image training requires CUDA. CPU supports model/loss/inference tests only.')
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed and os.name == 'nt':
        raise RuntimeError('Dual-GPU NCCL training requires Linux, e.g. Kaggle or WSL2.')
    if config['workers'] < 0 or config['workers'] > 4:
        raise ValueError('workers is the TOTAL loader-worker budget and must be between 0 and 4.')
    if config['image_size'] != 2048:
        raise ValueError('This experiment trains native 2048x2048 images; it never resizes.')
    if not 0 < config['memory_fraction'] < 1:
        raise ValueError('memory_fraction must be between zero and one.')
    if config['image_size'] % 2**int(config['depth']):
        raise ValueError('image_size must be divisible by 2**depth.')
    if config['gradient_accumulation_steps'] < 1:
        raise ValueError('gradient_accumulation_steps must be positive.')
    if config['epochs'] < 1 or config['patience'] < 1:
        raise ValueError('epochs and patience must be positive.')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(1)
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch.backends.cudnn.benchmark = bool(config['cudnn_benchmark'])
    if distributed:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    try:
        properties = torch.cuda.get_device_properties(device)
        physical_gpu = dict(rank=rank, local_rank=local_rank, name=properties.name,
                            total_vram_gib=properties.total_memory / 1024**3,
                            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', 'all'))
        physical_gpus = _gather_objects(physical_gpu, distributed, world_size)
        if rank == 0:
            print(f'Native2048 training | {world_size} GPU ranks | PyTorch {torch.__version__} '
                  f'| CUDA {torch.version.cuda}', flush=True)
            for gpu in physical_gpus:
                print(f'Rank {gpu["rank"]}: {gpu["name"]}, {gpu["total_vram_gib"]:.2f} GiB VRAM', flush=True)
            print('AMP + checkpointed encoder/decoder stages; whole-image loss and prediction. '
                  'DDP VRAM is per GPU. Accumulated Dice averages microbatch objectives.', flush=True)
        model_config = {key: config[key] for key in ('base_channels', 'depth', 'max_channels', 'checkpoint_blocks')}
        model_config.update(in_channels=1, out_channels=1)
        module = FullImageUNet(**model_config).to(device)
        if config['channels_last']:
            module.to(memory_format=torch.channels_last)
        module.train()
        selected_batch, probe_reports = probe_batch_sizes(module, config, device, distributed, rank, world_size)
        config['per_gpu_batch_size'] = selected_batch
        # Synchronize only after the unwrapped probes have passed on every card.
        model = DistributedDataParallel(module, device_ids=[local_rank],
                                       output_device=local_rank, broadcast_buffers=False,
                                       gradient_as_bucket_view=True) if distributed else module
        workers = config['workers'] // world_size + int(rank < config['workers'] % world_size)
        train_dataset = FullImageDataset(config['train_records'], augment=True, image_size=config['image_size'])
        val_dataset = FullImageDataset(config['val_records'], image_size=config['image_size'])
        if not len(train_dataset) or not len(val_dataset):
            raise ValueError('Training and validation image records must both be nonempty.')
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                           seed=config['seed'], drop_last=False) if distributed else None
        loader_options = dict(batch_size=selected_batch, pin_memory=True,
                              worker_init_fn=seed_loader_worker)
        train_options = dict(num_workers=workers)
        if workers:
            train_options.update(prefetch_factor=config['prefetch_factor'],
                                 persistent_workers=True, multiprocessing_context='spawn')
        generator = torch.Generator().manual_seed(config['seed'] + rank)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None,
                                  drop_last=False, generator=generator, **loader_options, **train_options)
        # No padding or repeated validation images; ranks may have different batch counts.
        val_loader = DataLoader(val_dataset, sampler=list(range(rank, len(val_dataset), world_size)),
                                num_workers=0, **loader_options)
        optimizer = _new_optimizer(module, config)
        scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
        if rank == 0:
            for key in ('checkpoint_path', 'history_path', 'runtime_report_path'):
                Path(config[key]).parent.mkdir(parents=True, exist_ok=True)
            print(f'Loaders: {config["workers"]} TOTAL training workers, validation workers=0; '
                  f'per-GPU batch={selected_batch}; nominal effective batch='
                  f'{selected_batch * world_size * config["gradient_accumulation_steps"]}', flush=True)
        best_selection_loss, stale_epochs, history, memory_history = float('inf'), 0, [], []
        for epoch in range(1, config['epochs'] + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            torch.cuda.reset_peak_memory_stats(device)
            train_metrics, train_seconds, train_rate = run_epoch(
                model, module, train_loader, optimizer, scaler, config, device,
                distributed, rank, world_size, True, epoch, config['channels_last'])
            val_metrics, val_seconds, val_rate = run_epoch(
                model, module, val_loader, optimizer, scaler, config, device,
                distributed, rank, world_size, False, epoch, config['channels_last'])
            free, total = torch.cuda.mem_get_info(device)
            memory = dict(rank=rank, peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 1024**3,
                          free_gib=free / 1024**3, total_gib=total / 1024**3)
            per_rank_memory = _gather_objects(memory, distributed, world_size)
            selection_loss = val_metrics['selection_loss']
            if not math.isfinite(selection_loss):
                raise RuntimeError('Validation selection loss is not finite; inspect inputs and AMP stability.')
            scheduler.step(selection_loss)
            improved = selection_loss < best_selection_loss - 1e-5
            if improved:
                best_selection_loss, stale_epochs = selection_loss, 0
            else:
                stale_epochs += 1
            row = dict(epoch=epoch)
            for split, metrics in (('train', train_metrics), ('val', val_metrics)):
                row.update({f'{split}_{name}': metrics[name] for name in (
                    'loss', 'log_soft_pq_loss', 'soft_dice', 'bce_loss', 'dice_at_0_5')})
            row.update(val_selection_loss=selection_loss, learning_rate=optimizer.param_groups[0]['lr'],
                       train_seconds=train_seconds, val_seconds=val_seconds,
                       train_images_per_second=train_rate, val_images_per_second=val_rate,
                       max_peak_allocated_gib=max(m['peak_allocated_gib'] for m in per_rank_memory),
                       max_peak_reserved_gib=max(m['peak_reserved_gib'] for m in per_rank_memory))
            if rank == 0:
                history.append(row)
                memory_history.append(dict(epoch=epoch, per_rank=per_rank_memory))
                if improved:
                    checkpoint_payload = dict(model_state_dict=module.state_dict(), model_config=model_config,
                                              image_size=config['image_size'], epoch=epoch,
                                              loss_weights=config['loss_weights'],
                                              num_gpus=world_size, per_gpu_batch_size=selected_batch,
                                              gradient_accumulation_steps=config['gradient_accumulation_steps'],
                                              channels_last=config['channels_last'], amp=config['amp'])
                    checkpoint_payload.update({f'val_{name}': value for name, value in val_metrics.items()})
                    torch.save(checkpoint_payload, config['checkpoint_path'])
                with open(config['history_path'], 'w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(history)
                _write_json(config['runtime_report_path'], dict(
                    model_config=model_config, image_size=config['image_size'], physical_gpus=physical_gpus,
                    world_size=world_size, total_training_workers=config['workers'], validation_workers=0,
                    channels_last=config['channels_last'], amp=config['amp'],
                    per_gpu_batch_size=selected_batch, gradient_accumulation_steps=config['gradient_accumulation_steps'],
                    memory_fraction=config['memory_fraction'], probe_reports=probe_reports,
                    epoch_memory=memory_history))
                print(f'Epoch {epoch:02d} | train loss {train_metrics["loss"]:.4f} | '
                      f'val loss {val_metrics["loss"]:.4f}, Dice@0.5 {val_metrics["dice_at_0_5"]:.4f} | '
                      f'{train_rate:.2f} images/s | peak allocated/reserved '
                      f'{row["max_peak_allocated_gib"]:.2f}/{row["max_peak_reserved_gib"]:.2f} GiB | '
                      f'{"saved best" if improved else "no improvement"}', flush=True)
            if distributed:
                dist.barrier(device_ids=[local_rank])
            if stale_epochs >= config['patience']:
                break
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    train(json.loads(Path(arguments.config).read_text(encoding='utf-8')))
