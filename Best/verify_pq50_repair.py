"""Sustained real-image, native-crop FP32 regression run; not a leaderboard benchmark."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import pq50_runtime as r


def run(updates=256):
    out = Path(__file__).resolve().parent / 'repair_2026_11_04' / 'validation'
    out.mkdir(parents=True, exist_ok=True)
    cfg = r.default_config()
    specs = copy.deepcopy(cfg['dense_models'])
    cfg.update(mode='smoke', gpu_ids=[0], output_root=str(out), run_name='numerics_native', smoke_sources=4)
    cfg, manifest = r.prepare(cfg)
    device = torch.device('cuda:0')
    records = []
    for member, spec in enumerate(specs):
        r.seed_all(20261104 + member)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        started = time.time()
        model = r.build_dense(spec, pretrained=True).to(device)
        r.train_mode(model)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = r.dense_optimizer(model, cfg)
        dataset = r.CropDataset(manifest['splits']['train'], spec['crop'], updates, cfg['seed'])
        loader = DataLoader(dataset, batch_size=1, num_workers=0)
        losses, foreground_crops, count = [], 0, 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            foreground_crops += int(y[:, 0].any())
            for group in optimizer.param_groups:
                group['lr'] = cfg['dense_lr'] * group['lr_scale'] * min(1., (step+1)/16)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=False):
                logits = model(x)
                loss = r.dense_loss(logits, y)
            loss.backward()
            count = r.dense_step(model, optimizer, ema, count, cfg['ema_decay'])
            losses.append(float(loss.detach()))
            if (step+1) % 32 == 0:
                print(json.dumps(dict(encoder=spec['encoder'], update=count, loss=losses[-1],
                                      mean_last32=float(np.mean(losses[-32:])))), flush=True)
        # Real full-disk validation image, production-sized tiles, no resized masks.
        image = cv2.imread(manifest['splits']['calibration'][0]['cache'])
        probability = r.dense_predict(ema, image, device, tile=1024, overlap=256, batch=1, tta=4)
        quantized = r.quantize_probabilities(probability)
        r.checkpoint_save(out / f'finite_native_{member}.pt', dict(state=ema.state_dict(), spec=spec,
            precision='float32', numerics_version=2, test_only=True, updates=count))
        assert count == updates and all(np.isfinite(losses))
        record = dict(encoder=spec['encoder'], native_crop=spec['crop'], batch=1, precision='float32',
            optimizer_updates=count, foreground_crops=foreground_crops,
            first32_loss=float(np.mean(losses[:32])), last32_loss=float(np.mean(losses[-32:])),
            all_losses_finite=True, native_prediction_shape=list(probability.shape),
            probability_min=float(probability.min()), probability_max=float(probability.max()),
            prediction_finite=True, quantization_valid=True, checkpoint_finite=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, seconds=time.time()-started)
        records.append(record)
        r.save_json(out / 'sustained_fp32.json', dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            models=records, note='Numerical regression test on four train images, not a PQ or leaderboard result.'))
        print(json.dumps(record), flush=True)
        del model, ema, optimizer, x, y, logits, loss, probability, quantized
        gc.collect(); torch.cuda.empty_cache()
    return records


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--updates', type=int, default=256)
    args = p.parse_args(); run(args.updates)
