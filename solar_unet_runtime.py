"""Self-contained U-Net worker embedded in solar-filament-u-net.ipynb for Kaggle."""
import argparse
import csv
import gc
import hashlib
import json
import os
import random
import shutil
import sys
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
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class CustomUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2, 2)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, 2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, 2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, 2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, 2)
        self.dec1 = DoubleConv(c * 2, c)
        self.output = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        x = self.bottleneck(self.pool(s4))
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        return self.output(x)


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
    """Combine the union-PQ surrogate, global foreground Dice, and pixel BCE.

    sync_dice pools sufficient statistics over all training ranks. The straight-
    through gradient is scaled by world size to compensate DDP's gradient mean.
    Validation must leave it off: ranks can have unequal numbers of batches.
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
        positive_loss = -0.5 * (
            soft_sq.clamp_min(1e-6).log() + soft_rq.clamp_min(1e-6).log()
        )
        empty_loss = predicted_area / float(targets.shape[-2] * targets.shape[-1])
        pq_loss = torch.where(target_area > 0, positive_loss, empty_loss).mean()
        # Reuse the overlap reductions; Dice pools pixels rather than averaging tiles.
        overlap = torch.stack((intersection.sum(), predicted_area.sum(), target_area.sum()))
        dice_overlap = overlap
        if sync_dice:
            global_overlap = overlap.detach().clone()
            dist.all_reduce(global_overlap)
            dice_overlap = global_overlap + dist.get_world_size() * (overlap - overlap.detach())
        soft_dice = (2.0 * dice_overlap[0] + smooth) / (
            dice_overlap[1] + dice_overlap[2] + smooth
        )
        # Unweighted BCE encourages calibration without reweighting class costs.
        # It also corrects confident mistakes when sigmoid-overlap gradients saturate.
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss = (
            weights['log_soft_pq'] * pq_loss
            + weights['dice'] * (1.0 - soft_dice)
            + weights['bce'] * bce_loss
        )
        return dict(loss=loss, log_soft_pq_loss=pq_loss, soft_dice=soft_dice,
                    bce_loss=bce_loss, overlap=overlap)


def log_soft_panoptic_quality_loss(logits, targets, smooth=1.0):
    """Retain the original surrogate as a separately logged diagnostic."""
    return segmentation_loss(logits, targets, smooth=smooth)['log_soft_pq_loss']


def summarize_epoch_totals(totals, weights, training):
    """Reduce validation Dice once over the complete, non-padding dataset."""
    samples = totals['samples']
    if samples == 0:
        raise ValueError('Empty epoch. Reduce the per-GPU batch or check metadata.')
    pq_loss = totals['log_soft_pq_loss'] / samples
    bce_loss = totals['bce_loss'] / samples
    if training:
        soft_dice = totals['soft_dice'] / samples
        hard_dice = float('nan')
        loss = totals['loss'] / samples
        selection_loss = float('nan')
    else:
        soft_dice = (2.0 * totals['soft_intersection'] + 1.0) / (
            totals['predicted_area'] + totals['target_area'] + 1.0
        )
        hard_dice = (2.0 * totals['hard_intersection'] + 1.0) / (
            totals['hard_denominator'] + 1.0
        )
        loss = (weights['log_soft_pq'] * pq_loss + weights['dice'] * (1.0 - soft_dice)
                + weights['bce'] * bce_loss)
        # Select against the same hard threshold shown in the epoch logs.
        selection_loss = (weights['log_soft_pq'] * pq_loss + weights['dice'] * (1.0 - hard_dice)
                          + weights['bce'] * bce_loss)
    return dict(loss=loss, log_soft_pq_loss=pq_loss, soft_dice=soft_dice,
                bce_loss=bce_loss, dice_at_0_5=hard_dice, selection_loss=selection_loss)


class FilamentTileDataset(Dataset):
    def __init__(self, manifest, augment=False):
        self.manifest = manifest
        self.tile_names = manifest['tile_names']
        self.augment = augment
        self.packed = None

    def __len__(self):
        return len(self.tile_names)

    def __getitem__(self, index):
        packed_path = self.manifest.get('packed_path')
        if packed_path:
            if self.packed is None:
                self.packed = np.load(packed_path, mmap_mode='r')
            image, mask = self.packed[index]
        else:
            root = Path(self.manifest['split_dir'])
            name = self.tile_names[index]
            image = cv2.imread(str(root / 'images' / name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(root / 'masks' / name), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise OSError(f'Unreadable tile pair: {root / name}')
            mask = (mask > 127).astype(np.uint8)
        if self.augment:
            if random.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if random.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)
            turns = random.randrange(4)
            if turns:
                image, mask = np.rot90(image, turns), np.rot90(mask, turns)
        # Copy read-only mmap views before exposing them as writable Torch tensors.
        image = np.array(image[None], dtype=np.uint8, order='C', copy=True)
        mask = np.array(mask[None], dtype=np.uint8, order='C', copy=True)
        return torch.from_numpy(image), torch.from_numpy(mask)


def build_tile_manifest(metadata, split_dir, cache_dir, tile_size, use_cache=True):
    """Pack once; all GPU processes and workers share OS-backed read-only pages."""
    split_dir, cache_dir = Path(split_dir), Path(cache_dir)
    tile_names = metadata['tile_name'].astype(str).tolist()
    manifest = dict(tile_names=tile_names, split_dir=str(split_dir.resolve()))
    if not use_cache:
        return manifest
    cache_dir.mkdir(parents=True, exist_ok=True)
    packed_path = cache_dir / f'{split_dir.name}.npy'
    signature_path = cache_dir / f'{split_dir.name}.json'
    metadata_stat = (split_dir / 'metadata.csv').stat()
    signature = hashlib.sha256(json.dumps([tile_size, tile_names, metadata_stat.st_mtime_ns, metadata_stat.st_size]).encode()).hexdigest()
    if packed_path.is_file() and signature_path.is_file():
        info = json.loads(signature_path.read_text(encoding='utf-8'))
        if info.get('signature') == signature:
            manifest['packed_path'] = str(packed_path.resolve())
            print(f'Reusing packed {split_dir.name} tiles: {len(tile_names):,}', flush=True)
            return manifest
    required = len(tile_names) * 2 * tile_size * tile_size + 1024
    if shutil.disk_usage(cache_dir).free < required + 1024**3:
        print(f'Insufficient disk for packed {split_dir.name} tiles; using PNG loader.', flush=True)
        return manifest
    started = time.perf_counter()
    temporary = packed_path.with_suffix('.partial.npy')
    packed = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.uint8, shape=(len(tile_names), 2, tile_size, tile_size))
    try:
        for index, name in enumerate(tile_names):
            image = cv2.imread(str(split_dir / 'images' / name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(split_dir / 'masks' / name), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape != (tile_size, tile_size) or mask.shape != image.shape:
                raise OSError(f'Missing or incorrectly sized tile pair: {split_dir / name}')
            packed[index, 0] = image
            packed[index, 1] = mask > 127
        packed.flush()
    finally:
        del packed
    os.replace(temporary, packed_path)
    signature_path.write_text(json.dumps(dict(signature=signature)), encoding='utf-8')
    manifest['packed_path'] = str(packed_path.resolve())
    print(f'Packed {len(tile_names):,} {split_dir.name} grayscale tile pairs in {time.perf_counter() - started:.1f}s; shared across GPU ranks.', flush=True)
    return manifest


def seed_loader_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


class CUDAPrefetcher:
    def __init__(self, loader, device=None, channels_last=False):
        self.loader = iter(loader)
        self.device = device or torch.device('cuda:0')
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=self.device)
        self.next_images = self.next_targets = None
        self.last_loader_wait_ms = 0.0
        self._preload()

    def _preload(self):
        started = time.perf_counter()
        try:
            images, targets = next(self.loader)
        except StopIteration:
            self.next_images = self.next_targets = None
            self.last_loader_wait_ms = 1000 * (time.perf_counter() - started)
            return
        self.last_loader_wait_ms = 1000 * (time.perf_counter() - started)
        with torch.cuda.stream(self.stream):
            images = images.to(self.device, dtype=torch.float32, non_blocking=True).div_(255)
            if self.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            self.next_images = images
            self.next_targets = targets.to(self.device, dtype=torch.float32, non_blocking=True)

    def __iter__(self):
        current = torch.cuda.current_stream(self.device)
        while self.next_images is not None:
            current.wait_stream(self.stream)
            images, targets = self.next_images, self.next_targets
            images.record_stream(current)
            targets.record_stream(current)
            self._preload()
            yield images, targets


def choose_layout(module, config, device, distributed, rank):
    """Compare layouts on the target GPU; restore BatchNorm and RNG state afterward."""
    if not config.get('autotune_layout', True):
        return False, []
    saved_buffers = {name: value.detach().clone() for name, value in module.named_buffers()}
    cpu_rng = torch.get_rng_state()
    gpu_rng = torch.cuda.get_rng_state(device)
    measurements = []
    size, batch = config['tile_size'], config['per_gpu_batch_size']
    images = torch.rand((batch, 1, size, size), device=device)
    targets = (torch.rand_like(images) > 0.97).float()
    try:
        for channels_last in (False, True):
            memory_format = torch.channels_last if channels_last else torch.contiguous_format
            module.to(memory_format=memory_format)
            inputs = images.contiguous(memory_format=memory_format)
            elapsed = []
            failed = 0
            try:
                for step in range(config.get('layout_warmup_steps', 2) + config.get('layout_measure_steps', 3)):
                    module.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                        logits = module(inputs)
                    loss = segmentation_loss(logits, targets, config.get('loss_weights'))['loss']
                    loss.backward()
                    torch.cuda.synchronize(device)
                    if step >= config.get('layout_warmup_steps', 2):
                        elapsed.append(1000 * (time.perf_counter() - started))
            except torch.cuda.OutOfMemoryError:
                failed = 1
            finally:
                module.zero_grad(set_to_none=True)
            score = float(np.median(elapsed)) if elapsed and not failed else float('inf')
            score_tensor = torch.tensor(score, device=device, dtype=torch.float64)
            if distributed:
                dist.all_reduce(score_tensor, op=dist.ReduceOp.MAX)
            score = score_tensor.item()
            measurements.append({'channels_last': channels_last, 'forward_backward_ms': score})
            if rank == 0:
                print(f'Layout trial: channels_last={channels_last}; slowest GPU forward+backward={score:.1f} ms', flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        module.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                buffer.copy_(saved_buffers[name])
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng, device)
    best = min(measurements, key=lambda row: row['forward_backward_ms'])
    if not np.isfinite(best['forward_backward_ms']):
        raise RuntimeError('Both layouts ran out of VRAM. Lower PER_GPU_BATCH_SIZE.')
    selected = best['channels_last']
    module.to(memory_format=torch.channels_last if selected else torch.contiguous_format)
    return selected, measurements


def reduce_metrics(values, device, distributed):
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor)
    return tensor.cpu().tolist()


def initialize_gradient_buffers(module):
    """Keep exact parameter strides, including ambiguous single-channel kernels."""
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.empty_strided(
                    parameter.size(), parameter.stride(),
                    dtype=parameter.dtype, device=parameter.device,
                ).zero_()


def run_epoch(model, module, loader, optimizer, scaler, config, device,
              distributed, rank, world_size, training, epoch, channels_last):
    model.train(training)
    if not training and distributed:
        # Validation uses the unwrapped model, with identical rank-0 BN statistics.
        # This also permits unequal final validation batches without DDP collectives.
        for buffer in module.buffers():
            dist.broadcast(buffer, src=0)
    prediction_model = model if training else module
    total_names = ('loss', 'log_soft_pq_loss', 'bce_loss', 'soft_dice',
                   'soft_intersection', 'predicted_area', 'target_area',
                   'hard_intersection', 'hard_denominator', 'samples')
    totals = {name: torch.zeros((), dtype=torch.float64, device=device) for name in total_names}
    weights = config.get('loss_weights', DEFAULT_LOSS_WEIGHTS)
    timing_events, loader_waits = [], []
    profile = training and epoch == 1 and config.get('profile', True)
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.inference_mode
    prefetch = CUDAPrefetcher(loader, device, channels_last)
    with context():
        for batch_index, (images, targets) in enumerate(prefetch, start=1):
            measure = profile and config['profile_warmup'] < batch_index <= config['profile_warmup'] + config['profile_steps']
            events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if measure else None
            if training:
                # Preserve DDP bucket views and the explicit singleton-dimension strides.
                optimizer.zero_grad(set_to_none=False)
            if events:
                events[0].record()
            with torch.autocast('cuda', dtype=torch.float16, enabled=config['amp']):
                logits = prediction_model(images)
            if batch_index == 1 and epoch == 1 and rank == 0:
                print(f'First {"train" if training else "validation"} batch: logits={logits.dtype}; input={tuple(images.shape)}; channels_last={channels_last}', flush=True)
            if events:
                events[1].record()
            components = segmentation_loss(logits, targets, weights, sync_dice=training and distributed)
            loss = components['loss']
            if events:
                events[2].record()
            if training:
                scaler.scale(loss).backward()
                if events:
                    events[3].record()
                scaler.step(optimizer)
                scaler.update()
                if events:
                    events[4].record()
                    timing_events.append(events)
                    loader_waits.append(prefetch.last_loader_wait_ms)
            for name in ('loss', 'log_soft_pq_loss', 'bce_loss', 'soft_dice'):
                totals[name] += components[name].detach().double() * images.shape[0]
            if not training:
                for name, value in zip(('soft_intersection', 'predicted_area', 'target_area'), components['overlap']):
                    totals[name] += value.detach().double()
                predictions, truth = logits.detach().ge(0), targets.bool()
                totals['hard_intersection'] += torch.logical_and(predictions, truth).sum()
                totals['hard_denominator'] += predictions.sum() + truth.sum()
            totals['samples'] += images.shape[0]
            if measure and batch_index == config['profile_warmup'] + config['profile_steps']:
                torch.cuda.synchronize(device)
                durations = np.mean([[events[i].elapsed_time(events[i + 1]) for i in range(4)] for events in timing_events], axis=0)
                profile_tensor = torch.tensor(list(durations) + [float(np.mean(loader_waits)), torch.cuda.max_memory_allocated(device) / 1024**3], device=device)
                if distributed:
                    profiles = [torch.zeros_like(profile_tensor) for _ in range(world_size)]
                    dist.all_gather(profiles, profile_tensor)
                else:
                    profiles = [profile_tensor]
                if rank == 0:
                    for gpu_id, values in enumerate(profiles):
                        forward, loss_ms, backward, optim_ms, wait_ms, memory = values.tolist()
                        print(f'GPU {gpu_id} real-step profile: forward {forward:.1f} ms | combined loss {loss_ms:.2f} ms | backward incl. gradient sync {backward:.1f} ms | optimizer {optim_ms:.1f} ms | loader wait {wait_ms:.1f} ms | peak {memory:.2f} GiB', flush=True)
            if rank == 0 and batch_index % config.get('log_every', 20) == 0:
                # One synchronization per progress report, never every training step.
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                throughput = batch_index * config['per_gpu_batch_size'] * world_size / elapsed
                print(f'  {"train" if training else "val"} {batch_index}/{len(loader)} batches | {elapsed:.1f}s | approx {throughput:.1f} tiles/s', flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    reduced = reduce_metrics(torch.stack(list(totals.values())), device, distributed)
    totals = dict(zip(total_names, reduced))
    metrics = summarize_epoch_totals(totals, weights, training)
    elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed = elapsed_tensor.item()
    return metrics, elapsed, totals['samples'] / elapsed


def train(config):
    config = dict(config)
    config['loss_weights'] = validate_loss_weights(config.get('loss_weights', DEFAULT_LOSS_WEIGHTS))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = config.get('cudnn_benchmark', False)
    if distributed:
        dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=device)
    try:
        if rank == 0:
            print(f'Persistent training processes: {world_size}; PyTorch {torch.__version__}; CUDA {torch.version.cuda}; cuDNN {torch.backends.cudnn.version()}; cudnn.benchmark={torch.backends.cudnn.benchmark}', flush=True)
            print(f'Loss weights: {config["loss_weights"]}; checkpoint selection uses hard Dice at 0.5 in place of soft Dice.', flush=True)
        workers = config['workers'] // world_size
        train_dataset = FilamentTileDataset(config['train'], augment=True)
        val_dataset = FilamentTileDataset(config['val'])
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, seed=config['seed'], drop_last=True) if distributed else None
        # An exact, non-padding validation partition prevents duplicated metrics.
        val_indices = list(range(rank, len(val_dataset), world_size))
        loader_options = dict(batch_size=config['per_gpu_batch_size'], pin_memory=True, worker_init_fn=seed_loader_worker)
        training_options = dict(num_workers=workers)
        if workers:
            training_options.update(persistent_workers=True, prefetch_factor=config['prefetch_factor'], multiprocessing_context='spawn')
        generator = torch.Generator().manual_seed(config['seed'] + rank)
        train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None, drop_last=True, generator=generator, **loader_options, **training_options)
        # Keep only four training workers alive across both ranks; validation reads mmap directly.
        val_loader = DataLoader(val_dataset, sampler=val_indices, num_workers=0, **loader_options)
        if not len(train_loader):
            raise ValueError('Not enough training tiles for the configured per-GPU batch.')
        if rank == 0:
            print(f'Loaders ready: {workers} workers/rank, {workers * world_size} total; {len(train_loader)} train batches/rank; global batch={config["per_gpu_batch_size"] * world_size}', flush=True)
        module = CustomUNet(base_channels=config['base_channels']).to(device)
        module.train()
        channels_last, layout_trials = choose_layout(module, config, device, distributed, rank)
        model = DistributedDataParallel(module, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, gradient_as_bucket_view=True, static_graph=True) if distributed else module
        initialize_gradient_buffers(module)
        kwargs = dict(lr=config['learning_rate'], weight_decay=config['weight_decay'])
        try:
            optimizer = torch.optim.Adam(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            optimizer = torch.optim.Adam(model.parameters(), **kwargs)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
        scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
        best_selection_loss, stale_epochs, history = float('inf'), 0, []
        for epoch in range(1, config['epochs'] + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if rank == 0:
                print(f'\nEpoch {epoch:02d}/{config["epochs"]} started', flush=True)
            train_metrics, train_seconds, train_rate = run_epoch(model, module, train_loader, optimizer, scaler, config, device, distributed, rank, world_size, True, epoch, channels_last)
            val_metrics, val_seconds, _ = run_epoch(model, module, val_loader, optimizer, scaler, config, device, distributed, rank, world_size, False, epoch, channels_last)
            selection_loss = val_metrics['selection_loss']
            scheduler.step(selection_loss)
            improved = selection_loss < best_selection_loss - 1e-5
            if improved:
                best_selection_loss, stale_epochs = selection_loss, 0
            else:
                stale_epochs += 1
            row = dict(epoch=epoch)
            for split, metrics in (('train', train_metrics), ('val', val_metrics)):
                for name in ('loss', 'log_soft_pq_loss', 'soft_dice', 'bce_loss', 'dice_at_0_5'):
                    row[f'{split}_{name}'] = metrics[name]
            row.update(val_selection_loss=selection_loss, learning_rate=optimizer.param_groups[0]['lr'], train_seconds=train_seconds, val_seconds=val_seconds, train_tiles_per_second=train_rate)
            if rank == 0:
                history.append(row)
                if improved:
                    checkpoint = dict(model_state_dict=module.state_dict(), epoch=epoch,
                                      loss_weights=config['loss_weights'], tile_size=config['tile_size'],
                                      base_channels=config['base_channels'], num_gpus=world_size,
                                      per_gpu_batch_size=config['per_gpu_batch_size'], channels_last=channels_last)
                    checkpoint.update({f'val_{name}': value for name, value in val_metrics.items()})
                    torch.save(checkpoint, config['checkpoint_path'])
                with open(config['history_path'], 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(history)
                Path(config['runtime_report_path']).write_text(json.dumps(dict(channels_last=channels_last, layout_trials=layout_trials, world_size=world_size), indent=2), encoding='utf-8')
                print(f'Epoch {epoch:02d} | train loss {train_metrics["loss"]:.4f} | val loss {val_metrics["loss"]:.4f}, soft-PQ loss {val_metrics["log_soft_pq_loss"]:.4f}, Dice {val_metrics["dice_at_0_5"]:.4f}, BCE {val_metrics["bce_loss"]:.4f}, selection loss {selection_loss:.4f} | train {train_seconds:.1f}s ({train_rate:.1f} tiles/s), val {val_seconds:.1f}s | {"saved best" if improved else "no improvement"}', flush=True)
            if distributed:
                dist.barrier(device_ids=[local_rank])
            if stale_epochs >= config['patience']:
                if rank == 0:
                    print('Early stopping.', flush=True)
                break
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text(encoding='utf-8')))
