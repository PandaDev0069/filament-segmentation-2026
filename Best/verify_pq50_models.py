"""Optional actual-model CPU-DDP / CUDA crop checks, separate from the notebook."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import pq50_runtime as r


def ddp_worker(rank, init, target):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    model = r.build_dense(dict(encoder="resnet18"), pretrained=False)
    r.train_mode(model)
    wrapper = DistributedDataParallel(model, broadcast_buffers=False)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = r.dense_optimizer(model, r.default_config())
    updates = 0
    for step in range(2):
        torch.manual_seed(100 + rank + step * 5)
        x = torch.rand(2, 3, 64, 64)
        y = torch.zeros(2, 2, 64, 64)
        y[:, 0, 15 + rank:20 + rank, 15:40] = 1
        optimizer.zero_grad(set_to_none=True)
        loss = r.dense_loss(wrapper(x), y)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        updates = r.dense_step(model, optimizer, ema, updates, .995, synchronize=True)
    checksum = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    hashes = [None, None]
    dist.all_gather_object(hashes, checksum)
    assert len(set(hashes)) == 1
    # A fault on only rank one must stop EVERY rank at the same point.
    stopped = False
    try:
        r.require_finite(torch.tensor(float('nan') if rank == 1 else 1.), 'injected rank-one failure', synchronize=True)
    except FloatingPointError:
        stopped = True
    stopped_ranks = [None, None]
    dist.all_gather_object(stopped_ranks, stopped)
    assert all(stopped_ranks)
    if rank == 0:
        r.save_json(target, dict(two_cpu_ranks=True, identical_post_update_parameters=True,
            backward_steps=updates, loss=float(loss.detach()), parameter_sha=checksum,
            one_rank_nan_stops_both=True,
            limitation="Gloo CPU exercises the actual model graph, not two-T4 CUDA/NCCL execution."))
    dist.destroy_process_group()


def check_cuda(target, pretrained=False, batch=1):
    device = torch.device("cuda:0")
    results = []
    for spec in r.default_config()["dense_models"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = r.build_dense(spec, pretrained).to(device)
        r.train_mode(model)
        size = spec["crop"]
        x = torch.rand(batch, 3, size, size, device=device)
        y = torch.zeros(batch, 2, size, size, device=device)
        y[:, 0, size//2:size//2 + 5, size//2:size//2 + 70] = 1
        optimizer = r.dense_optimizer(model, r.default_config())
        for group in optimizer.param_groups:
            group['lr'] *= group['lr_scale']
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        before = next(model.parameters()).detach().clone()
        with torch.autocast("cuda", enabled=False):
            logits = model(x)
            loss = r.dense_loss(logits, y)
        assert logits.shape == y.shape and torch.isfinite(loss)
        loss.backward()
        unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not unused, unused
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert r.dense_step(model, optimizer, ema, 0, .995) == 1
        torch.cuda.synchronize()
        assert not torch.equal(before, next(model.parameters()))
        result = dict(encoder=spec["encoder"], crop=size, batch=batch, pretrained=pretrained, precision='float32',
            finite_backward_and_adam_update=True, loss=float(loss.detach()),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        print(json.dumps(result), flush=True)
        results.append(result)
        r.save_json(target, dict(gpu=torch.cuda.get_device_name(0), dense_models=results))
        del model, ema, x, y, optimizer, logits, loss, before
        import gc
        gc.collect()
    r.save_json(target, dict(gpu=torch.cuda.get_device_name(0), dense_models=results))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["ddp", "cuda"], required=True)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()
    output = Path(__file__).resolve().parent / "pq50_validation"
    output.mkdir(exist_ok=True)
    if args.task == "ddp":
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            mp.spawn(ddp_worker, args=((Path(tmp) / "init").as_uri(), str(output / "cpu_ddp.json")), nprocs=2, join=True)
        print((output / "cpu_ddp.json").read_text())
    else:
        check_cuda(output / "cuda_model_checks.json", args.pretrained, args.batch)
