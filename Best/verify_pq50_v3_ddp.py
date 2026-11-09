"""Two-rank CPU check of new model/loss gradient participation and synchronization."""
import importlib.util
import json
import os
import uuid
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

HERE = Path(__file__).resolve().parent

def run_rank(rank, store_path):
    spec = importlib.util.spec_from_file_location("v3_ddp_check", HERE/"pq50_v3_runtime.py")
    r = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r)
    dist.init_process_group("gloo", init_method=Path(store_path).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    try:
        rank = dist.get_rank()
        torch.manual_seed(17+rank)
        model = r.build_dense(dict(encoder="resnet18", crop=64, refine=True), pretrained=False)
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        r.train_mode(model)
        optimizer = r.dense_optimizer(model, r.default_config())
        observed = []
        for step in range(2):
            image = torch.randn(1, 3, 64, 64)
            target = torch.zeros(1, 3, 64, 64)
            target[:, 0, 20:30, 15:45] = 1
            target[:, 2, 10:40, 5:55] = 1
            optimizer.zero_grad(set_to_none=True)
            loss = r.dense_loss(wrapped(image), target)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            optimizer.step()
            vector = torch.cat([model.refine[-1].weight.detach().flatten(), model.core.segmentation_head[0].weight.detach().flatten()])
            gathered = [torch.zeros_like(vector) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, vector)
            assert all(torch.equal(gathered[0], x) for x in gathered[1:])
            observed.append(float(loss.detach()))
        if rank==0:
            result = dict(kind="two-rank CPU Gloo/DDP", iterations=2, all_trainable_parameters_have_gradients=True,
                synchronized_parameters=True, losses_on_rank0=observed, dual_cuda_training_verified=False)
            r.save_json(HERE/"analysis_2026_11_08"/"v3_ddp_result.json", result)
            print(json.dumps(result))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    store_path = HERE/"analysis_2026_11_08"/("ddp_store_"+uuid.uuid4().hex)
    try:
        torch.multiprocessing.spawn(run_rank, args=(str(store_path),), nprocs=2, join=True)
    finally:
        store_path.unlink(missing_ok=True)
