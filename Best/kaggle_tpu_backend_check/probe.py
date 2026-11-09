"""Short backend validation only: no competition images, labels or predictions."""
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
OUT = Path("/kaggle/working/tpu_backend_check")
OUT.mkdir(exist_ok=True)


def worker(index):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    from torch import nn
    from torch.nn import functional as F

    started = time.time()
    torch.set_num_threads(1)
    torch.manual_seed(20261109)
    device = torch_xla.device()
    rank, world = xr.global_ordinal(), xr.world_size()
    if xr.device_type() != "TPU" or world != 8:
        raise RuntimeError(f"Expected eight TPU workers, got {xr.device_type()} / {world}")
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
                          nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
                          nn.Conv2d(32, 16, 3, padding=1), nn.GELU(),
                          nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False),
                          nn.Conv2d(16, 1, 1)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    generator = torch.Generator().manual_seed(400 + rank)
    image = torch.randn((2, 3, 256, 256), generator=generator).to(device)
    target = torch.zeros((2, 1, 256, 256))
    target[:, :, 120 + rank:128 + rank, 30:220] = 1
    target = target.to(device)
    losses = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("xla", dtype=torch.bfloat16):
            logits = model(image)
        logits = logits.float()
        probability = logits.sigmoid()
        loss = F.binary_cross_entropy_with_logits(logits, target)
        loss = loss + (1 - (2 * (probability * target).sum() + 1) /
                       (probability.sum() + target.sum() + 1))
        loss.backward()
        xm.optimizer_step(optimizer)
        torch_xla.sync()
        losses.append(float(loss.cpu()))
        if not math.isfinite(losses[-1]):
            raise FloatingPointError("Non-finite TPU loss")
    parameters = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    gradients = torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters()])
    if not torch.isfinite(parameters).all() or not torch.isfinite(gradients).all():
        raise FloatingPointError("Non-finite TPU parameters or gradients")
    row = dict(rank=rank, world=world, device_type=xr.device_type(), losses=losses,
               parameters=float(parameters.double().sum()),
               parameter_square_sum=float(parameters.double().square().sum()),
               finite_gradients=True, seconds=time.time()-started, precision="bfloat16 convolution / float32 loss")
    (OUT / f"rank_{rank}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    started = time.time()
    versions = {}
    for package in ("torch", "torch-xla", "torchvision", "libtpu", "torch-tpu", "jax"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    inventory = dict(packages=versions, cpu_count=os.cpu_count(),
                     memory=Path("/proc/meminfo").read_text().splitlines()[:2],
                     competition_data_used=False, model_quality_measured=False)
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print(json.dumps(inventory), flush=True)
    import torch_xla
    torch_xla.launch(worker, args=())
    rows = [json.loads((OUT / f"rank_{rank}.json").read_text()) for rank in range(8)]
    for key in ("parameters", "parameter_square_sum"):
        values = [row[key] for row in rows]
        if max(values)-min(values) > 1e-5:
            raise RuntimeError("TPU gradient reduction did not synchronize parameters")
    result = dict(seconds=time.time()-started, workers=8, parameter_synchronization_verified=True,
                  finite_gradients_all_workers=True, model_quality_measured=False)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print("TPU backend validation complete:", json.dumps(result), flush=True)
