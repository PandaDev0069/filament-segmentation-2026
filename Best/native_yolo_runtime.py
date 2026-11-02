"""Native-image YOLO26 segmentation runtime, also embedded in the companion notebook.

Ultralytics 8.4.198, Python >=3.10. No input-image resampling or polygon conversion.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import csv
import gc
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from ultralytics import YOLO, settings
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import LOCAL_RANK, RANK
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss, v8SegmentationLoss
from ultralytics.utils.torch_utils import torch_distributed_zero_first, strip_optimizer
from ultralytics.utils import ops

SIZE = 2048
VERSION = "native-yolo26-union-v1"
cv2.setNumThreads(1)
torch.set_num_threads(1)
settings.update({k: False for k in ("wandb", "mlflow", "clearml", "comet", "tensorboard", "raytune") if k in settings})


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_map(function, items, workers=4):
    """Bound both threads and pending results; exceptions propagate to the caller."""
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(function, item))
        while pending:
            yield pending.popleft().result()
            item = next(iterator, None)
            if item is not None:
                pending.append(pool.submit(function, item))


def assert_native(array):
    shape = tuple(array.shape[-2:]) if torch.is_tensor(array) else tuple(array.shape[:2])
    if shape != (SIZE, SIZE):
        raise ValueError(f"Expected native {SIZE}x{SIZE}, got {shape}. Resizing is forbidden.")


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    assert_native(image)
    return image


def preprocess(image, variant, cfg):
    assert_native(image)
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Preprocessing expects native uint8 grayscale")
    if variant == "raw":
        result = image.copy()
    elif variant == "clahe":
        result = cv2.createCLAHE(clipLimit=cfg["clahe_clip"],
                                tileGridSize=tuple(cfg["clahe_grid"])).apply(image)
    elif variant == "percentile":
        # Ignore the black image background when estimating intensity bounds.
        pixels = image[image > 0]
        low, high = np.percentile(pixels, cfg["percentiles"]) if pixels.size else (0, 0)
        result = (np.clip((image.astype(np.float32) - low) * 255 / (high - low), 0, 255)
                  .astype(np.uint8)) if high > low else image.copy()
    elif variant == "gamma":
        table = np.round(255 * (np.arange(256) / 255) ** cfg["gamma"]).astype(np.uint8)
        result = cv2.LUT(image, table)
    else:
        raise ValueError(f"Unknown preprocessing variant: {variant}")
    assert_native(result)
    return np.ascontiguousarray(result)


def encode_mask(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(mask.shape), "counts": rle["counts"].decode("ascii")}


def decode_mask(rle):
    return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)


def annotation_mask(annotation):
    segmentation = annotation["segmentation"]
    if isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((SIZE, SIZE), dtype=bool)
        for polygon in segmentation:
            xy = np.asarray(polygon, dtype=float)
            if xy.ndim != 1 or len(xy) < 6 or len(xy) % 2 or not np.isfinite(xy).all():
                raise ValueError(f"Invalid polygon: {annotation['id']}")
            xy = xy.reshape(-1, 2)
            if (xy < 0).any() or (xy > SIZE).any():
                raise ValueError(f"Polygon outside native image: {annotation['id']}")
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, SIZE, SIZE))
    elif isinstance(segmentation, dict):
        rle = dict(segmentation)
        if rle.get("size") != [SIZE, SIZE]:
            raise ValueError("RLE size differs from native image")
        if isinstance(rle["counts"], list):
            if any(not isinstance(v, int) or v < 0 for v in rle["counts"]) or sum(rle["counts"]) != SIZE * SIZE:
                raise ValueError("Invalid uncompressed RLE")
            rle = mask_utils.frPyObjects(rle, SIZE, SIZE)
        elif isinstance(rle["counts"], str):
            rle["counts"] = rle["counts"].encode("ascii")
    else:
        raise ValueError("Unsupported annotation representation")
    mask = mask_utils.decode(rle).astype(bool)
    assert_native(mask)
    return mask


def union_labels(masks, min_area=1, connectivity=8):
    union = np.zeros((SIZE, SIZE), np.uint8)
    for mask in masks:
        assert_native(mask)
        union |= mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(union, connectivity=connectivity)
    keep = sorted((i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area),
                  key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), int(stats[i, 1]), int(stats[i, 0])))
    lut = np.zeros(count, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    labels = lut[labels]
    boxes = []
    for i in keep:
        x, y, w, h = (int(v) for v in stats[i, :4])
        boxes.append([(x + w / 2) / SIZE, (y + h / 2) / SIZE, w / SIZE, h / SIZE])
    return labels, boxes


def find_root(configured=""):
    name = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    candidates = [Path(configured).expanduser()] if configured else []
    if not candidates:
        for ancestor in (Path.cwd(), *Path.cwd().parents):
            candidates.extend([ancestor, ancestor / "MAGFilo_1.0_Kaggle_2026"])
        if Path("/kaggle/input").exists():
            candidates.extend(p.parent.parent for p in Path("/kaggle/input").rglob(name))
    for root in candidates:
        if (root / "train" / name).is_file() and (root / "test/test_images").is_dir():
            return root.resolve()
    raise FileNotFoundError("Set data_root to the directory containing train/ and test/.")


def prepare_data(cfg, run):
    root = find_root(cfg["data_root"])
    annotation_path = root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    payload = load_json(annotation_path)
    if len({str(a["id"]) for a in payload["annotations"]}) != len(payload["annotations"]):
        raise ValueError("Duplicate annotation IDs")
    images = {str(i["id"]): i for i in payload["images"]}
    if len(images) != len(payload["images"]):
        raise ValueError("Duplicate image IDs")
    grouped, annotations = defaultdict(list), defaultdict(list)
    for annotation in payload["annotations"]:
        if str(annotation["image_id"]) not in images or annotation.get("iscrowd", 0):
            raise ValueError("Unknown image or unsupported crowd annotation")
        annotations[str(annotation["image_id"])].append(annotation)
    for image in images.values():
        if (image["height"], image["width"]) != (SIZE, SIZE):
            raise ValueError(f"Non-native metadata: {image['file_name']}")
        grouped[image["file_name"]].append(str(image["id"]))
    train_dir = (root / "train/train_images").resolve()

    def index_source(item):
        name, ids = item
        path = (train_dir / name).resolve()
        if Path(name).name != name or not path.is_relative_to(train_dir):
            raise ValueError(f"Invalid source filename: {name}")
        pixels = read_gray(path)
        return {"name": name, "path": str(path), "image_ids": ids,
                "pixel_hash": hashlib.sha256(pixels.tobytes()).hexdigest()}

    records = list(ordered_map(index_source, sorted(grouped.items()), cfg["cpu_workers"]))
    hashes = sorted({r["pixel_hash"] for r in records})
    if len(hashes) < 2:
        raise ValueError("Need at least two independent sources")
    np.random.default_rng(cfg["seed"]).shuffle(hashes)
    n_val = min(len(hashes) - 1, max(1, round(len(hashes) * cfg["val_fraction"])))
    validation = set(hashes[:n_val])
    partitions = {"train": [r for r in records if r["pixel_hash"] not in validation],
                  "val": [r for r in records if r["pixel_hash"] in validation]}
    if cfg["mode"] == "smoke":
        # Include an actual multi-annotator source in the smoke training partition.
        partitions["train"].sort(key=lambda r: (len(r["image_ids"]) == 1, r["name"]))
        partitions = {k: v[:cfg["smoke_sources"]] for k, v in partitions.items()}
    key = digest({"version": VERSION, "annotations": file_sha(annotation_path), "seed": cfg["seed"],
                  "val_fraction": cfg["val_fraction"], "merge": cfg["merge"], "mode": cfg["mode"],
                  "sources": records, "active": {k: [r["name"] for r in v] for k, v in partitions.items()}})
    target_dir = Path(run) / "targets" / key[:16]
    target_dir.mkdir(parents=True, exist_ok=True)

    def merge_source(record):
        record = dict(record)
        target = target_dir / (digest(record["name"])[:24] + ".npz")
        masks = (annotation_mask(a) for image_id in record["image_ids"] for a in annotations[image_id])
        labels, boxes = union_labels(masks, **cfg["merge"])
        np.savez_compressed(target, labels=labels)
        record.update(target=str(target), boxes=boxes, foreground_pixels=int(np.count_nonzero(labels)))
        return record

    for partition in partitions:
        partitions[partition] = list(ordered_map(merge_source, partitions[partition], cfg["cpu_workers"]))
    test_dir = root / "test/test_images"
    test_paths = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if len({p.stem for p in test_paths}) != len(test_paths):
        raise ValueError("Test stems must be unique for submission IDs")
    if cfg["mode"] == "smoke":
        test_paths = test_paths[:cfg["smoke_sources"]]
    manifest = {"key": key, "root": str(root), "annotation_path": str(annotation_path),
                **partitions, "test": [{"name": p.name, "path": str(p)} for p in test_paths]}
    manifest_path = Path(run) / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path)["key"] != key:
        raise RuntimeError("Dataset/split changed. Use a fresh run_name.")
    save_json(manifest_path, manifest)
    save_json(Path(run) / "data_audit.json", {
        "native_shape": [SIZE, SIZE], "source_images": len(records), "annotator_records": len(images),
        "multi_annotator_sources": sum(len(r["image_ids"]) > 1 for r in records),
        "pixel_identical_groups": len(records) - len(hashes), "dataset_key": key,
        "active_sources": {k: len(v) for k, v in partitions.items()},
        "active_instances": {k: sum(len(r["boxes"]) for r in v) for k, v in partitions.items()},
        "merge_policy": cfg["merge"], "validation_policy": "original annotator records; averaged per source"})
    return manifest_path


class NativeDataset(Dataset):
    """Read original pixels and exact native instance-ID maps; no YOLO polygon rasterization."""
    rect = False
    mosaic = False

    def __init__(self, records, cfg, variant, training=False):
        self.records, self.cfg, self.variant, self.training = records, cfg, variant, training
        self.im_files = [r["path"] for r in records]
        self.labels = [{"im_file": r["path"], "shape": (SIZE, SIZE),
                        "cls": np.zeros((len(r["boxes"]), 1), np.float32),
                        "bboxes": np.asarray(r["boxes"], np.float32).reshape(-1, 4),
                        "segments": [], "bbox_format": "xywh", "normalized": True} for r in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = preprocess(read_gray(record["path"]), self.variant, self.cfg["preprocessing"])
        with np.load(record["target"]) as target:
            labels = target["labels"].copy()
        assert_native(labels)
        boxes = self.labels[index]["bboxes"].copy()
        if self.training:
            if random.random() < self.cfg["fliplr"]:
                image, labels = np.fliplr(image), np.fliplr(labels)
                boxes[:, 0] = 1 - boxes[:, 0]
            if random.random() < self.cfg["flipud"]:
                image, labels = np.flipud(image), np.flipud(labels)
                boxes[:, 1] = 1 - boxes[:, 1]
            gain = random.uniform(1 - self.cfg["brightness"], 1 + self.cfg["brightness"])
            image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        rgb = np.repeat(image[None], 3, axis=0)
        return {"img": torch.from_numpy(np.ascontiguousarray(rgb)),
                "masks": torch.from_numpy(np.ascontiguousarray(labels[None])),
                "cls": torch.zeros((len(boxes), 1), dtype=torch.float32),
                "bboxes": torch.from_numpy(boxes), "batch_idx": torch.zeros(len(boxes)),
                "im_file": record["path"], "ori_shape": (SIZE, SIZE), "resized_shape": (SIZE, SIZE),
                "ratio_pad": ((1.0, 1.0), (0.0, 0.0))}

    @staticmethod
    def collate_fn(samples):
        result = {key: tuple(s[key] for s in samples) for key in samples[0]}
        result["img"] = torch.stack(result["img"])
        for key in ("masks", "cls", "bboxes"):
            result[key] = torch.cat(result[key])
        result["batch_idx"] = torch.cat([s["batch_idx"] + i for i, s in enumerate(samples)])
        return result


def native_anchor_loss(coefficient, proto, label_map, mask_id, box, area):
    """Upsample predictions, never targets. Checkpointed per anchor to bound memory."""
    logits = (coefficient.float() @ proto.float().flatten(1)).reshape(1, 1, *proto.shape[-2:])
    logits = F.interpolate(logits, (SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0]
    target = (label_map == mask_id).float()
    # Match Ultralytics' normalized crop-mask BCE, including fractional box edges.
    x = torch.arange(SIZE, device=logits.device)[None]
    y = torch.arange(SIZE, device=logits.device)[:, None]
    inside = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (bce * inside).mean() / area.clamp_min(1 / (SIZE * SIZE))


class NativeMaskLoss(v8SegmentationLoss):
    """Full native mask supervision; standard box assignment avoids target downsampling."""
    def get_assigned_targets_and_loss(self, preds, batch):
        return v8DetectionLoss.get_assigned_targets_and_loss(self, preds, batch)

    def loss(self, preds, batch):
        assert_native(batch["masks"])
        coefficients = preds["mask_coefficient"].permute(0, 2, 1).contiguous()
        prototype = preds["proto"]
        prototype, semantic = prototype if isinstance(prototype, tuple) else (prototype, None)
        (fg, indices, boxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        seg = prototype.float().sum() * 0 + coefficients.float().sum() * 0
        sem = seg * 0
        count = fg.sum().clamp_min(1)
        for image_index in range(len(prototype)):
            for anchor in torch.where(fg[image_index])[0]:
                box = boxes[image_index, anchor].detach()
                area = ((box[2] - box[0]) * (box[3] - box[1])) / (SIZE * SIZE)
                args = (coefficients[image_index, anchor], prototype[image_index], batch["masks"][image_index],
                        indices[image_index, anchor] + 1, box, area)
                seg = seg + (checkpoint(native_anchor_loss, *args, use_reentrant=False)
                             if torch.is_grad_enabled() else native_anchor_loss(*args))
        if semantic is not None:
            sem_logits = F.interpolate(semantic.float(), (SIZE, SIZE), mode="bilinear", align_corners=False)
            sem = self.bcedice_loss(sem_logits, (batch["masks"] > 0).float().unsqueeze(1))
        losses = torch.stack((det_loss[0], seg / count * self.hyp.box, det_loss[1], det_loss[2], sem * self.hyp.box))
        return losses * len(prototype), dict(zip(self.loss_names, losses.detach()))


class NativeSegmentationModel(SegmentationModel):
    def init_criterion(self):
        return (E2ELoss(self, NativeMaskLoss) if getattr(self.model[-1], "one2one_cv2", None) is not None
                else NativeMaskLoss(self))


class NativeValidator(SegmentationValidator):
    def preprocess(self, batch):
        assert_native(batch["img"])
        assert_native(batch["masks"])
        return super().preprocess(batch)

    def init_metrics(self, model):
        super().init_metrics(model)
        self.process = ops.process_mask_native

    def _prepare_batch(self, si, batch):
        result = DetectionValidator._prepare_batch(self, si, batch)
        n = result["cls"].shape[0]
        ids = torch.arange(1, n + 1, device=batch["masks"].device).view(n, 1, 1)
        result["masks"] = (batch["masks"][si] == ids).float()
        return result


class NativeTrainer(SegmentationTrainer):
    def get_dataset(self):
        self.job = load_json(self.args.data)
        self.native_cfg = self.job["cfg"]
        self.manifest = load_json(self.job["manifest"])
        return {"train": "train", "val": "val", "nc": 1, "names": {0: "filament"}, "channels": 3}

    def build_dataset(self, img_path, mode="train", batch=None):
        return NativeDataset(self.manifest[mode], self.native_cfg, self.job["variant"], mode == "train")

    def get_dataloader(self, dataset_path, batch_size=1, rank=-1, mode="train"):
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        # No second persistent validation worker pool; total training workers <= cpu_workers.
        return build_dataloader(dataset, batch=1 if mode == "val" else batch_size,
                                workers=self.args.workers if mode == "train" else 0,
                                shuffle=mode == "train", rank=rank, device=self.device)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = NativeSegmentationModel(cfg, nc=1, ch=3, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch):
        assert_native(batch["img"])
        assert_native(batch["masks"])
        batch = super().preprocess_batch(batch)
        rank = max(RANK, 0)
        audit = self.save_dir / f"native_tensor_rank{rank}.json"
        if not audit.exists():
            save_json(audit, {"image_tensor": list(batch["img"].shape), "mask_tensor": list(batch["masks"].shape),
                              "rank": rank, "device": str(self.device), "target_downsampling": False})
        return batch

    def _setup_train(self):
        # Avoid the SDK's unrelated resized 640px AMP example. Actual native AMP is checked by smoke training.
        import ultralytics.engine.trainer as trainer_module
        previous = trainer_module.check_amp
        trainer_module.check_amp = lambda model: self.device.type == "cuda"
        try:
            return super()._setup_train()
        finally:
            trainer_module.check_amp = previous

    def get_validator(self):
        return NativeValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def final_eval(self):
        # Standalone SDK validation would reload a stock polygon dataset. Epoch validation uses NativeValidator;
        # the notebook evaluates best.pt on original annotator masks after training on both GPUs.
        with torch_distributed_zero_first(LOCAL_RANK):
            if RANK in (-1, 0):
                for path in (self.last, self.best):
                    if path.exists():
                        strip_optimizer(path)


def train_job(job_path):
    job = load_json(job_path)
    cfg = job["cfg"]
    local_gpu = max(LOCAL_RANK, 0)
    torch.cuda.set_device(local_gpu)
    total = torch.cuda.get_device_properties(local_gpu).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.9, cfg["vram_budget_gib"] * 2**30 / total), local_gpu)
    gpu_count = len(cfg["gpu_ids"])
    args = dict(data=str(Path(job_path).resolve()), model=job.get("resolved_weights", cfg["weights"]), imgsz=SIZE,
                epochs=1 if cfg["mode"] == "smoke" else cfg["epochs"], patience=cfg["patience"],
                batch=gpu_count * cfg["batch_per_gpu"], device=",".join(str(i) for i in range(gpu_count)),
                workers=cfg["cpu_workers"] // gpu_count, cache=False, max_det=cfg["max_det"],
                project=str(Path(job["run"]).parent), name=Path(job["run"]).name, exist_ok=True,
                pretrained=True, optimizer="AdamW", lr0=cfg["lr0"], lrf=0.01, cos_lr=True,
                seed=cfg["seed"], deterministic=True, amp=cfg["amp"], nbs=cfg["nominal_batch"],
                overlap_mask=True, mask_ratio=1, rect=False, multi_scale=0.0,
                degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                fliplr=cfg["fliplr"], flipud=cfg["flipud"], hsv_h=0.0, hsv_s=0.0, hsv_v=cfg["brightness"],
                mosaic=0.0, close_mosaic=0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                plots=cfg["plots"], verbose=True)
    if cfg.get("resume"):
        args["resume"] = str(Path(job["run"]) / "weights/last.pt")
    trainer = NativeTrainer(overrides=args)
    trainer.train()


def prediction_path(job, category, record):
    return Path(job["run"]) / "predictions" / category / (digest(record["name"])[:24] + ".json")


def predict_job(job_path):
    job = load_json(job_path)
    cfg = job["cfg"]
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(0.9, cfg["vram_budget_gib"] * 2**30 / total), 0)
    model = YOLO(job["checkpoint"])
    key = digest({"weights": file_sha(job["checkpoint"]), "variant": job["variant"],
                  "cfg": cfg["preprocessing"], "data": load_json(job["manifest"])["key"],
                  "conf": min(cfg["confidences"]), "max_det": cfg["max_det"], "version": VERSION})

    def read(record):
        return preprocess(read_gray(record["path"]), job["variant"], cfg["preprocessing"])

    def pack(record, masks, confidence):
        result = {"key": key, "name": record["name"], "source_sha": file_sha(record["path"]),
                  "instances": [{"rle": encode_mask(mask), "confidence": float(c)} for mask, c in zip(masks, confidence)]}
        save_json(prediction_path(job, job["category"], record), result)

    records = job["records"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_read = pool.submit(read, records[0]) if records else None
        pending_write = None
        for index, record in enumerate(records):
            image = pending_read.result()
            if index + 1 < len(records):
                pending_read = pool.submit(read, records[index + 1])
            path = prediction_path(job, job["category"], record)
            if path.exists():
                cached = load_json(path)
                if cached.get("key") == key and cached.get("source_sha") == file_sha(record["path"]):
                    continue
            # Tensor sources bypass all SDK letterboxing and input resizing.
            tensor = torch.from_numpy(np.repeat(image[None, None], 3, axis=1)).to("cuda:0").float() / 255
            assert_native(tensor)
            with torch.inference_mode():
                result = model.predict(tensor, imgsz=SIZE, conf=min(cfg["confidences"]), iou=0.7,
                                       device=0, retina_masks=True, max_det=cfg["max_det"],
                                       quantize=16 if cfg["amp"] else None, augment=False, verbose=False)[0]
            if result.masks is None:
                masks, scores = [], []
            else:
                assert_native(result.masks.data)
                masks = result.masks.data.detach().cpu().numpy().astype(bool)
                scores = result.boxes.conf.detach().cpu().numpy()
            if pending_write:
                pending_write.result()
            pending_write = pool.submit(pack, record, masks, scores)
            print(f"{job['category']} {index + 1}/{len(records)} {record['name']}: {len(scores)} instances", flush=True)
            del tensor, result
        if pending_write:
            pending_write.result()


def run_commands(commands, log_dir, environment=None):
    """Stream bounded log tails, stop peers on failure/interrupt, and reap every process."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    processes, handles, readers = [], [], []
    try:
        for name, command, extra_env in commands:
            handle = (log_dir / f"{name}.log").open("w", encoding="utf-8")
            handles.append(handle)
            process = subprocess.Popen(command, env={**os.environ, **(environment or {}), **extra_env},
                                       stdout=handle, stderr=subprocess.STDOUT)
            processes.append((name, process))
            readers.append((log_dir / f"{name}.log").open("r", encoding="utf-8", errors="replace"))
        while True:
            for (name, _), reader in zip(processes, readers):
                output = reader.read(64 * 1024)
                if output.strip():
                    print(f"[{name}] {output}", end="", flush=True)
            failed = [(name, p.returncode) for name, p in processes if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Worker failed: {failed}. See {log_dir}")
            if all(p.poll() == 0 for _, p in processes):
                break
            time.sleep(0.5)
    finally:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for (name, _), reader in zip(processes, readers):
            tail = reader.read()
            if tail.strip():
                print(f"[{name}] {tail}", end="", flush=True)
            reader.close()
        for handle in handles:
            handle.close()


def resolve_training_weights(configured):
    """Resolve/download once in the coordinator, before any DDP ranks can read the file."""
    path = Path(configured).expanduser()
    if not path.is_file():
        from ultralytics.utils.downloads import attempt_download_asset
        path = Path(attempt_download_asset(str(configured)))
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Pretrained checkpoint could not be resolved: {configured}")
    return path.resolve()


def launch_training(job_path, runtime_path):
    job = load_json(job_path)
    cfg = job["cfg"]
    # Both ranks otherwise call the SDK downloader on the same destination concurrently.
    checkpoint = (Path(job["run"]) / "weights/last.pt") if cfg.get("resume") else cfg["weights"]
    weights = resolve_training_weights(checkpoint)
    job["resolved_weights"] = str(weights)
    job["resolved_weights_sha256"] = file_sha(weights)
    save_json(job_path, job)
    visible = ",".join(str(i) for i in cfg["gpu_ids"])
    command = [sys.executable, "-u"]
    if len(cfg["gpu_ids"]) > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={len(cfg['gpu_ids'])}"]
    command += [str(runtime_path), "train", str(job_path)]
    run_commands([("train", command, {"CUDA_VISIBLE_DEVICES": visible})], Path(job["run"]) / "logs")


def launch_prediction(job_path, runtime_path, category):
    job = load_json(job_path)
    manifest = load_json(job["manifest"])
    commands = []
    for rank, gpu in enumerate(job["cfg"]["gpu_ids"]):
        records = manifest[category][rank::len(job["cfg"]["gpu_ids"])]
        if not records:
            continue
        part = {**job, "records": records, "category": category,
                "checkpoint": str(Path(job["run"]) / "weights/best.pt")}
        part_path = Path(job["run"]) / f"{category}_gpu{rank}.json"
        save_json(part_path, part)
        commands.append((f"{category}_gpu{rank}", [sys.executable, "-u", str(runtime_path), "predict", str(part_path)],
                         {"CUDA_VISIBLE_DEVICES": str(gpu)}))
    run_commands(commands, Path(job["run"]) / "logs")


def final_components(instances, confidence, min_area):
    masks = (decode_mask(i["rle"]) for i in instances if i["confidence"] >= confidence)
    labels, _ = union_labels(masks, min_area=min_area)
    return [encode_mask(labels == i) for i in range(1, int(labels.max()) + 1)]


def instance_metrics(predicted, targets, iou_threshold=0.5):
    n, m = len(targets), len(predicted)
    if n == 0 and m == 0:
        return {"PQ": 1.0, "SQ": 1.0, "RQ": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if n and m:
        pred_rles = [{"size": r["size"], "counts": r["counts"].encode("ascii")} for r in predicted]
        gt_rles = [{"size": r["size"], "counts": r["counts"].encode("ascii")} for r in targets]
        iou = mask_utils.iou(gt_rles, pred_rles, [0] * m)
        eligible = iou > iou_threshold
        # Maximize match cardinality first, then matched IoU; allow overlapping annotator GT.
        rows, cols = linear_sum_assignment(-(eligible * (min(n, m) + 1) + iou * eligible))
        selected = eligible[rows, cols]
        tp = int(selected.sum())
        total_iou = float(iou[rows[selected], cols[selected]].sum())
    else:
        tp, total_iou = 0, 0.0
    fp, fn = m - tp, n - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {"PQ": total_iou / denominator if denominator else 1.0,
            "SQ": total_iou / tp if tp else 0.0, "RQ": tp / denominator if denominator else 1.0,
            "tp": tp, "fp": fp, "fn": fn}


def pixel_metrics(predicted, targets):
    a, b = np.zeros((SIZE, SIZE), bool), np.zeros((SIZE, SIZE), bool)
    for rle in predicted:
        a |= decode_mask(rle)
    for rle in targets:
        b |= decode_mask(rle)
    intersection = int(np.count_nonzero(a & b))
    size = int(a.sum()) + int(b.sum())
    union = int(np.count_nonzero(a | b))
    return {"Dice": 2 * intersection / size if size else 1.0, "IoU": intersection / union if union else 1.0}


def evaluate_job(job_path):
    import pandas as pd
    job = load_json(job_path)
    cfg, manifest = job["cfg"], load_json(job["manifest"])
    annotations = defaultdict(list)
    for annotation in load_json(manifest["annotation_path"])["annotations"]:
        annotations[str(annotation["image_id"])].append(annotation)

    def evaluate(record):
        cache_path = prediction_path(job, "val", record)
        cached = load_json(cache_path)["instances"]
        originals = [[encode_mask(annotation_mask(a)) for a in annotations[image_id]] for image_id in record["image_ids"]]
        rows = []
        for confidence in cfg["confidences"]:
            predicted = final_components(cached, confidence, cfg["prediction_min_area"])
            for image_id, targets in zip(record["image_ids"], originals):
                rows.append({"file_name": record["name"], "image_id": image_id, "confidence": confidence,
                             **instance_metrics(predicted, targets), **pixel_metrics(predicted, targets)})
        return rows

    rows = [row for result in ordered_map(evaluate, manifest["val"], cfg["cpu_workers"]) for row in result]
    per_record = pd.DataFrame(rows)
    per_record.to_csv(Path(job["run"]) / "validation_per_annotator.csv", index=False)
    # Equal weight per physical source, regardless of how many annotators labeled it.
    metrics = ["PQ", "SQ", "RQ", "Dice", "IoU", "tp", "fp", "fn"]
    per_source = per_record.groupby(["file_name", "confidence"], as_index=False)[metrics].mean()
    summary = per_source.groupby("confidence", as_index=False)[metrics].mean()
    summary.to_csv(Path(job["run"]) / "threshold_metrics.csv", index=False)
    per_source.to_csv(Path(job["run"]) / "validation_per_source.csv", index=False)
    best = summary.sort_values(["PQ", "Dice", "confidence"], ascending=[False, False, True]).iloc[0].to_dict()
    save_json(Path(job["run"]) / "validation_best.json", {"variant": job["variant"], **best})
    return {"variant": job["variant"], **best}


def create_submission(job_path, confidence):
    job = load_json(job_path)
    records = load_json(job["manifest"])["test"]
    name = "smoke_submission.csv" if job["cfg"]["mode"] == "smoke" else "submission.csv"
    destination = Path(job["run"]) / name
    coverage = []

    def encode_record(record):
        instances = load_json(prediction_path(job, "test", record))["instances"]
        predicted = final_components(instances, confidence, job["cfg"]["prediction_min_area"])
        return record["name"], [(f"{Path(record['name']).stem}_{i}", rle["counts"]) for i, rle in enumerate(predicted, 1)]

    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        seen, row_count = set(), 0
        for source, rows in ordered_map(encode_record, records, job["cfg"]["cpu_workers"]):
            for filament_id, rle in rows:
                if filament_id in seen or not rle:
                    raise ValueError("Duplicate ID or empty RLE")
                seen.add(filament_id)
                writer.writerow([filament_id, rle])
                row_count += 1
            coverage.append({"name": source, "instances": len(rows)})
    if len(coverage) != len(records):
        raise RuntimeError("Incomplete test source coverage")
    os.replace(temporary, destination)
    save_json(destination.with_suffix(".manifest.json"), {"mode": job["cfg"]["mode"], "variant": job["variant"],
              "confidence": confidence, "native_shape": [SIZE, SIZE], "checkpoint_sha": file_sha(Path(job["run"]) / "weights/best.pt"),
              "rows": row_count, "source_coverage": coverage})
    return destination


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("train", "predict"):
        raise SystemExit("Usage: native_yolo_runtime.py train|predict job.json")
    (train_job if sys.argv[1] == "train" else predict_job)(sys.argv[2])


if __name__ == "__main__":
    # Import once under a stable module name so saved models work in fresh DDP/prediction interpreters.
    import native_yolo_runtime
    native_yolo_runtime.main()
