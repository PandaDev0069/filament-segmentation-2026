"""Source replacements used by build_pq50_v3.py (embedded in the final notebook).

This file is build input, not an extra Kaggle input. The generated runtime contains
these definitions and the retained, audited I/O/normalization/scoring functions.
"""


def default_config():
    return dict(mode="full", data_root="", output_root="/kaggle/working",
        run_name="filament_instance_balanced_v3", gpu_ids=[0, 1], cpu_workers=4,
        seed=20261103, refit_full=False, calibration_fraction=.12, report_fraction=.12,
        minimum_free_gib=2., pretrained=True,
        dense_models=[dict(encoder="resnet34", crop=768, refine=True),
                      dict(encoder="tu-tf_efficientnetv2_s", crop=768, refine=True)],
        dense_epochs=60, dense_steps=240, dense_batch_per_gpu=2, auto_batch=True,
        maximum_batch_per_gpu=6, dense_lr=3e-4, encoder_lr_scale=.1,
        weight_decay=1e-4, accumulation=1, ema_decay=.99, dense_precision="float32",
        minimum_report_pq=.10, training_plots=True, pq_every=6,
        tile_size=1024, tile_overlap=256, inference_batch=2, tta=4,
        max_dense_candidates=192, quality_folds=5,
        pixel_thresholds=[.25, .35, .45, .55, .65], smoke_sources=3,
        total_hours=9.5, training_hours=5.6, owner_radius=16,
        use_previous_weights=True,
        previous_fingerprint="d7e1902f4a7c9cd244f438b8fe6f238fc669daec919224191e1843a945c0cabe",
        warm_start_checkpoints=[], previous_manifest="",
        refit_hours=1., refit_epochs=12, refit_steps=120,
        detector_conf=0., detector_weights="", detector_size=2048,
        detector_epochs=0, detector_batch_per_gpu=1, detector_lr=.001, max_det=100,
        session_started=time.time())


def deadline_remaining(cfg):
    return cfg["session_started"] + cfg["total_hours"] * 3600 - time.time()


def check_deadline(cfg, stage, reserve=0):
    remaining = deadline_remaining(cfg)
    if remaining < reserve:
        raise TimeoutError(f"{stage}: only {remaining/60:.1f} minutes remain in the "
                           "9.5-hour session budget. Existing artifacts are preserved.")
    return remaining


def configure_previous_weights(cfg):
    """Optionally reuse this user's exact, train-only previous run on Kaggle."""
    cfg = copy.deepcopy(cfg)
    if cfg["mode"] != "full" or not cfg.get("use_previous_weights", False):
        return cfg
    candidates = []
    for path in INPUT_ROOT.rglob("manifest.json"):
        try:
            value = load_json(path)
        except (OSError, ValueError):
            continue
        if value.get("config", {}).get("fingerprint") != cfg["previous_fingerprint"]:
            continue
        checkpoints = [path.parent/"selection_models"/f"dense_{i}"/"best.pt"
                       for i in range(len(cfg["dense_models"]))]
        if all(p.is_file() for p in checkpoints):
            candidates.append((path, checkpoints))
    if not candidates:
        print("Previous notebook output not attached: using generic ImageNet initialization.", flush=True)
        return cfg
    path, checkpoints = candidates[0]
    cfg.update(previous_manifest=str(path), warm_start_checkpoints=[str(p) for p in checkpoints],
        warm_start_sha256=[file_sha(p) for p in checkpoints], dense_lr=min(cfg["dense_lr"], 1.5e-4))
    print("Found previous train-only weights. Exact split/image checks will precede training.", flush=True)
    return cfg


def verify_previous_split(cfg, manifest):
    if not cfg.get("warm_start_checkpoints"):
        return
    previous = load_json(cfg["previous_manifest"])
    if previous["config"]["annotation_sha"] != cfg["annotation_sha"]:
        raise ValueError("Warm-start annotations differ from the current competition training data")
    for part in ("train", "calibration", "report"):
        old = {s["stem"]: s["image_sha"] for s in previous["splits"][part]}
        new = {s["stem"]: s["image_sha"] for s in manifest["splits"][part]}
        if old != new:
            raise ValueError(f"Warm-start {part} partition/image hashes differ: held-out contamination is possible")
    save_json(Path(cfg["run"])/"warm_start_audit.json", dict(
        previous_fingerprint=previous["config"]["fingerprint"], original_train_only_weights=True,
        identical_train_calibration_report_partitions=True, identical_image_hashes=True,
        annotation_sha=cfg["annotation_sha"], checkpoint_sha256=cfg["warm_start_sha256"]))
    print("Warm-start split, image hashes and annotation SHA verified.", flush=True)


def make_owner_map(rles, radius=16, shape=(2048, 2048)):
    """Nearest-instance assignment inside a narrow background collar.

    Original independent annotator masks remain the foreground reference. This
    auxiliary map gives each instance an equal soft-IoU term, including nearby
    background false positives. It never merges different annotators' masks.
    """
    owner = np.zeros(shape, np.uint16)
    best = np.full(shape, np.inf, np.float32)
    for index, rle in enumerate(rles, 1):
        if index >= 65535:
            raise ValueError("Too many instances for the auxiliary uint16 target")
        binary = decode(rle).astype(np.uint8)
        if binary.shape != shape:
            raise ValueError("Annotation dimensions do not match the image")
        ys, xs = np.nonzero(binary)
        if not len(xs):
            continue
        x0, x1 = max(0, int(xs.min()) - radius), min(shape[1], int(xs.max()) + radius + 1)
        y0, y1 = max(0, int(ys.min()) - radius), min(shape[0], int(ys.max()) + radius + 1)
        local = binary[y0:y1, x0:x1]
        distance = cv2.distanceTransform(1 - local, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        previous = best[y0:y1, x0:x1]
        update = (distance <= radius) & (distance < previous)
        owner[y0:y1, x0:x1][update] = index
        previous[update] = distance[update]
    return owner


def prepare_instance_targets(cfg, manifest, include_holdout=False):
    check_deadline(cfg, "instance target preparation", reserve=600)
    def process(source):
        for view in source["views"]:
            target = Path(cfg["run"]) / "cache" / (view["id"] + "_owner.png")
            if not target.is_file():
                owner = make_owner_map(view["rles"], cfg["owner_radius"])
                if not cv2.imwrite(str(target), owner):
                    raise OSError(target)
            view["owner"] = str(target)
        return source
    # Holdout targets are built only AFTER the validated model/report is frozen,
    # if a separate all-data refit has explicitly been enabled.
    parts = ("train", "calibration", "report") if include_holdout else ("train",)
    for part in parts:
        manifest["splits"][part] = list(bounded_map(process,
            manifest["splits"][part], cfg["cpu_workers"]))
    save_json(Path(cfg["run"]) / "manifest.json", manifest)
    print("Prepared independent instance targets; no external labels or test supervision.", flush=True)
    return manifest


def audit_split_leakage(cfg, manifest):
    """Assert physical dates and pixel-identical images cannot leak across splits."""
    rows = [(part, s) for part, sources in manifest["splits"].items() for s in sources]
    def hashed(row):
        part, source = row
        pixels = read_gray(source["image"])
        return dict(part=part, stem=source["stem"], date=source["date"],
                    decoded_sha256=hashlib.sha256(pixels.tobytes()).hexdigest())
    records = list(bounded_map(hashed, rows, cfg["cpu_workers"]))
    groups = defaultdict(list)
    for row in records:
        groups[row["decoded_sha256"]].append(row)
    collisions = [v for v in groups.values() if len({r["part"] for r in v})>1]
    training_dates = {s["date"] for s in manifest["splits"]["train"]}
    heldout_dates = {s["date"] for p in ("calibration", "report") for s in manifest["splits"][p]}
    save_json(Path(cfg["run"])/"split_leakage_audit.json", dict(
        decoded_images=len(records), cross_partition_pixel_duplicates=collisions,
        train_heldout_date_overlap=sorted(training_dates & heldout_dates), records=records))
    if collisions or training_dates & heldout_dates:
        raise ValueError("Cross-partition duplicate/date leakage detected; inspect split_leakage_audit.json")
    print(f"Split audit passed for {len(records)} decoded physical images.", flush=True)


class RefinedUNet(nn.Module):
    """ImageNet U-Net plus a full-resolution intensity/logit residual head."""
    def __init__(self, spec, pretrained=False):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.core = smp.Unet(encoder_name=spec["encoder"],
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3, classes=2, decoder_channels=(128, 64, 32, 16, 16),
            decoder_attention_type="scse")
        def replace_bn(module):
            for name, child in list(module.named_children()):
                if isinstance(child, nn.BatchNorm2d):
                    setattr(module, name, nn.GroupNorm(math.gcd(8, child.num_features), child.num_features))
                else:
                    replace_bn(child)
        replace_bn(self.core.decoder)
        self.core.decoder.blocks[-1].attention1.requires_grad_(False)
        self.refine = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.GroupNorm(4, 16),
            nn.SiLU(), nn.Conv2d(16, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 2, 1)) if spec.get("refine", True) else None
        if self.refine is not None:
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

    @property
    def encoder(self):
        return self.core.encoder

    @property
    def decoder(self):
        return self.core.decoder

    def forward(self, image):
        logits = self.core(image)
        return logits if self.refine is None else logits + self.refine(torch.cat([image, logits], 1))


def build_dense(spec, pretrained=False):
    return RefinedUNet(spec, pretrained)


class CropDataset(Dataset):
    def __init__(self, sources, crop, length, seed):
        self.sources, self.crop, self.length, self.seed, self.epoch = sources, crop, length, seed, 0
        self.cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * 10_000_019 + int(index))
        source = self.sources[int(rng.integers(len(self.sources)))]
        view = source["views"][int(rng.integers(len(source["views"])))]
        if view["id"] not in self.cache:
            self.cache.clear()
            image = cv2.imread(source["cache"])
            packed = cv2.imread(view["target"], cv2.IMREAD_GRAYSCALE)
            owner = cv2.imread(view["owner"], cv2.IMREAD_UNCHANGED)
            if image is None or packed is None or owner is None:
                raise OSError("Missing image/target/owner cache")
            self.cache[view["id"]] = image, packed, owner
        image, packed, owner = self.cache[view["id"]]
        # Source context varies; output pixel grids and test masks stay native-sized.
        size = min(SIZE, max(128, int(round(self.crop * rng.uniform(.70, 1.45) / 32)) * 32))
        if rng.random() < .5:
            size = self.crop  # retain exact original pixel spacing in half of crops
        choice = rng.random()
        if choice < .70 and view["boxes"]:
            boxes = np.asarray(view["boxes"])
            # Uniform instance sampling: small/thin filaments are not area-suppressed.
            x, y, w, h = boxes[int(rng.integers(len(boxes)))]
            cx, cy = x + w * rng.uniform(.2, .8), y + h * rng.uniform(.2, .8)
            x0, y0 = cx - size * rng.uniform(.3, .7), cy - size * rng.uniform(.3, .7)
        elif choice < .85:
            # Sample dark backgrounds inside the disk, rather than the black exterior.
            points = rng.integers(200, SIZE - 200, size=(48, 2))
            valid = ((points - SIZE/2)**2).sum(1) < (SIZE*.39)**2
            points = points[valid]
            if not len(points):
                points = np.array([[SIZE//2, SIZE//2]])
            values = image[points[:, 0], points[:, 1], 1].astype(np.float32)
            values += ((packed[points[:, 0], points[:, 1]] & 1) > 0) * 255
            cy, cx = points[int(np.argmin(values))]
            x0, y0 = cx - size/2, cy - size/2
        else:
            x0, y0 = rng.integers(0, SIZE - size + 1, size=2)
        x0, y0 = int(np.clip(x0, 0, SIZE-size)), int(np.clip(y0, 0, SIZE-size))
        x = image[y0:y0+size, x0:x0+size].copy()
        p = packed[y0:y0+size, x0:x0+size]
        ids = owner[y0:y0+size, x0:x0+size]
        if size != self.crop:
            interpolation = cv2.INTER_AREA if size > self.crop else cv2.INTER_LINEAR
            x = cv2.resize(x, (self.crop, self.crop), interpolation=interpolation)
            p = cv2.resize(p, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
            ids = cv2.resize(ids, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST)
        y = np.stack([(p & 1) > 0, (p & 2) > 0, ids], -1).astype(np.float32)
        k = int(rng.integers(4))
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < .5:
            x, y = x[:, ::-1], y[:, ::-1]
        x = x.astype(np.float32) / 255
        x = np.clip((x-.5) * rng.uniform(.8, 1.2) + .5 + rng.uniform(-.06, .06), 0, 1)
        if rng.random() < .25:
            x = x ** rng.uniform(.85, 1.15)
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(y.transpose(2, 0, 1))))


def instance_iou_loss(probability, foreground, owners):
    """Equal-weight soft IoU over visible instances, not over foreground pixels.

    Scatter reductions include the assigned local background collar in each union.
    This is a differentiable surrogate; it is not the official discrete PQ metric.
    """
    losses = []
    for p, y, ids in zip(probability, foreground, owners):
        ids = ids.flatten().long()
        p, y = p.flatten(), y.flatten()
        # IDs originate in train annotations and have a small bounded range.
        count = int(ids.max().detach().item()) + 1
        intersection = p.new_zeros(count).scatter_add(0, ids, p*y)
        predicted = p.new_zeros(count).scatter_add(0, ids, p)
        target = p.new_zeros(count).scatter_add(0, ids, y)
        valid = target > 0
        valid[0] = False
        value = 1 - (intersection + 1) / (predicted + target - intersection + 1)
        # Empty crops contribute zero here; BCE still supervises their background.
        losses.append((value * valid).sum() / valid.sum().clamp_min(1))
    return torch.stack(losses).mean()


def dense_loss(logits, targets, check=True):
    if targets.shape[1] != 3:
        raise ValueError("v3 requires foreground, contour and independent instance-owner targets")
    if check:
        require_finite(logits, "mask loss logits")
        require_finite(targets, "mask loss targets")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits, targets = logits.float(), targets.float()
        fg, edge, owner = targets[:, :1], targets[:, 1:2], targets[:, 2:3]
        probability = logits[:, :1].sigmoid()
        bce = (F.binary_cross_entropy_with_logits(logits[:, :1], fg, reduction="none") * (1+2*edge)).mean()
        axes = (1, 2, 3)
        dice = 1 - ((2*(probability*fg).sum(axes)+1)/(probability.sum(axes)+fg.sum(axes)+1)).mean()
        instance = instance_iou_loss(probability, fg, owner)
        contour = F.binary_cross_entropy_with_logits(logits[:, 1:2], edge,
            pos_weight=logits.new_tensor(3.))
        return .35*bce + .20*dice + .40*instance + .05*contour


def plan_steps(cfg, seconds_per_step, member, validation_seconds=0.):
    """Complete a cosine schedule inside a measured training allocation."""
    seconds = cfg["training_hours"] * 3600 / cfg.get("budget_members", len(cfg["dense_models"]))
    available = max(1., seconds - validation_seconds - 300)
    steps = max(8, int(available / max(seconds_per_step * 1.65 * cfg["dense_epochs"], .001)))
    return min(cfg["dense_steps"], steps)


def fixed_rule_counts(model, sources, device, cfg):
    counts = np.zeros(4, np.float64)
    for source in sources:
        image = cv2.imread(source["cache"])
        p = dense_predict(model, image, device, cfg["tile_size"], cfg["tile_overlap"], cfg["inference_batch"], 1)[0]
        rles = [full_rle(box, mask) for box, mask in components(p, .4, 80, cfg["max_dense_candidates"])
                if p[box[1]:box[3], box[0]:box[2]][mask].max() > .8]
        for view in source["views"]:
            counts += metric_counts(view["rles"], rles)
    return counts


def probe_batch(model, device, spec, cfg, world):
    """Measure real FP32 forward/backward; both ranks use the smaller safe batch."""
    initial = cfg["dense_batch_per_gpu"]
    candidates = [initial] if not cfg["auto_batch"] else list(range(cfg["maximum_batch_per_gpu"], 0, -1))
    model.eval()
    result, seconds = None, None
    for batch in candidates:
        x = y = z = loss = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            x = torch.zeros((batch, 3, spec["crop"], spec["crop"]), device=device)
            y = torch.zeros_like(x)
            y[:, 0, 40:80, 40:120] = 1
            y[:, 2, 30:90, 30:130] = 1
            timings = []
            for trial in range(3):
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                before = time.monotonic()
                z = model(x)
                loss = dense_loss(z, y)
                loss.backward()
                torch.cuda.synchronize(device)
                timings.append(time.monotonic()-before)
            peak = torch.cuda.max_memory_reserved(device)
            total = torch.cuda.get_device_properties(device).total_memory
            # Leave room for optimizer state, DDP buckets, EMA and actual variation.
            if peak > .74 * total and cfg["auto_batch"] and batch > 1:
                continue
            result, seconds = batch, float(np.median(timings[1:]))
            break
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
        finally:
            model.zero_grad(set_to_none=True)
            del x, y, z, loss
            gc.collect()
            torch.cuda.empty_cache()
    if result is None:
        raise RuntimeError("No native crop batch fit in the GPU")
    values = torch.tensor([float(result), float(seconds)], device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(values[:1], op=dist.ReduceOp.MIN)
        dist.all_reduce(values[1:], op=dist.ReduceOp.MAX)
    return int(values[0].item()), float(values[1].item())


def train_dense(config_path, member, full=False):
    cfg = load_json(config_path)
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    if full:
        cfg.update(dense_epochs=cfg["refit_epochs"] if cfg["mode"]=="full" else 1,
            dense_steps=cfg["refit_steps"] if cfg["mode"]=="full" else 2,
            training_hours=cfg["refit_hours"], dense_lr=cfg["dense_lr"]*.33,
            budget_members=len(selected_dense_ids(cfg)),
            warm_start_checkpoints=[str(Path(cfg["run"])/"selection_models"/f"dense_{i}"/"best.pt")
                                    for i in range(len(cfg["dense_models"]))])
        manifest["splits"]["train"] = (manifest["splits"]["train"]+
            manifest["splits"]["calibration"]+manifest["splits"]["report"])
    device, rank, world = device_setup(cfg, distributed=True)
    torch.backends.cudnn.benchmark = True
    seed_all(cfg["seed"] + member*101)
    spec = cfg["dense_models"][member]
    out = Path(cfg["run"]) / ("full_models" if full else "selection_models") / f"dense_{member}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").is_file():
        load_dense_checkpoint(out / "best.pt")
        return
    last = out / "last_training.pt"
    recovery = torch.load(last, map_location="cpu", weights_only=False) if last.is_file() else None
    if recovery is not None:
        require_finite_state(recovery)
        if recovery["fingerprint"] != cfg["fingerprint"]:
            raise RuntimeError("Resume fingerprint differs from the current run")
    warm = cfg.get("warm_start_checkpoints", [])
    if rank == 0:
        model = build_dense(spec, cfg["pretrained"] and recovery is None and not warm)
    if world > 1:
        dist.barrier()
    if rank != 0:
        model = build_dense(spec, cfg["pretrained"] and recovery is None and not warm)
    if recovery is None and warm:
        checkpoint = load_dense_checkpoint(warm[member])
        if checkpoint["spec"]["encoder"] != spec["encoder"]:
            raise ValueError("Warm-start encoder does not match")
        if any(key.startswith("core.") for key in checkpoint["state"]):
            model.load_state_dict(checkpoint["state"])
        else:
            model.core.load_state_dict(checkpoint["state"])
        del checkpoint
    model = model.to(device)
    if recovery is not None:
        batch, measured, steps = recovery["batch"], 0., recovery["steps"]
    elif cfg["mode"] == "smoke":
        batch, measured = cfg["dense_batch_per_gpu"], .1
        steps = cfg["dense_steps"]
    else:
        batch, measured = probe_batch(model, device, spec, cfg, world)
        # One calibration pass is timed with the actual model/device.
        validation_seconds, count = 0., 0
        if not full:
            model.eval()
            torch.cuda.synchronize(device)
            before = time.monotonic()
            image = cv2.imread(manifest["splits"]["calibration"][rank % len(manifest["splits"]["calibration"])] ["cache"])
            dense_predict(model, image, device, cfg["tile_size"], cfg["tile_overlap"], cfg["inference_batch"], 1)
            torch.cuda.synchronize(device)
            validation_seconds = (time.monotonic()-before) * math.ceil(len(manifest["splits"]["calibration"])/world)
            count = math.ceil(cfg["dense_epochs"]/cfg["pq_every"]) + 1 + int(bool(warm))
        steps = plan_steps(cfg, measured, member, validation_seconds*count)
        planned = torch.tensor(steps, device=device)
        if world > 1:
            dist.all_reduce(planned, op=dist.ReduceOp.MIN)
        steps = int(planned.item())
    resource = dict(member=member, gpu=str(device), gpu_name=torch.cuda.get_device_name(device),
        batch_per_gpu=batch, global_batch=batch*world, crop=spec["crop"], precision="float32",
        measured_forward_backward_seconds=measured, steps_per_epoch=steps,
        epochs=cfg["dense_epochs"], loader_workers=cfg["cpu_workers"]//world,
        requested_steps=cfg["dense_steps"], all_data_refit=full,
        training_sources=len(manifest["splits"]["train"]))
    save_json(out / f"rank_{rank}_resources.json", resource)
    print("Measured training plan:", json.dumps(resource), flush=True)
    sources = manifest["splits"]["train"]
    dataset = CropDataset(sources, spec["crop"], steps*batch*world, cfg["seed"]+member*111)
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=cfg["seed"])
    workers = cfg["cpu_workers"]//world
    options = dict(num_workers=workers, pin_memory=True, persistent_workers=False)
    if workers:
        options["prefetch_factor"] = 2
    loader = DataLoader(dataset, batch_size=batch, sampler=sampler, drop_last=True, **options)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = dense_optimizer(model, cfg)
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    begin, best, best_epoch, updates = 0, -1., 0, 0
    if recovery is not None:
        model.load_state_dict(recovery["model"])
        ema.load_state_dict(recovery["ema"])
        optimizer.load_state_dict(recovery["optimizer"])
        begin, best, best_epoch, updates = recovery["epoch"], recovery["best"], recovery["best_epoch"], recovery["updates"]
        del recovery
    elif world > 1:
        # DDP synchronizes random decoder/refiner initialization from rank zero.
        ema.load_state_dict(model.state_dict())
    training_start = time.time()
    allocation = cfg["training_hours"]*3600/cfg.get("budget_members", len(cfg["dense_models"]))
    reserve_hours = .65 if full else min(2.8, cfg["total_hours"]*.35)
    stop_at = min(training_start+allocation, cfg["session_started"]+(cfg["total_hours"]-reserve_hours)*3600)
    completed = begin
    if warm and begin == 0 and not full:
        counts = torch.tensor(fixed_rule_counts(ema, manifest["splits"]["calibration"][rank::world], device, cfg),
                              device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(counts)
        best = summarize(counts.cpu().numpy())["PQ"]
        if rank == 0:
            checkpoint_save(out/"best.pt", dict(state=ema.state_dict(), spec=spec, epoch=0,
                fingerprint=cfg["fingerprint"], precision="float32", numerics_version=3, checkpoint_PQ=best))
            print(f"Warm-start calibration PQ before fine-tuning: {best:.6f}. This checkpoint remains eligible.", flush=True)
    for epoch in range(begin, cfg["dense_epochs"]):
        started = time.time()
        dataset.epoch = epoch
        sampler.set_epoch(epoch)
        train_mode(model)
        total = 0.
        torch.cuda.reset_peak_memory_stats(device)
        for step, (x, y) in enumerate(loader):
            # Time progress also decays the LR when the measured budget is tight.
            progress = max((epoch+step/max(steps, 1))/cfg["dense_epochs"],
                           min(1., (time.time()-training_start)/max(stop_at-training_start, 1)))
            warmup = min(1., (epoch*steps+step+1)/max(steps*2, 1))
            lr = cfg["dense_lr"]*warmup*(.03+.97*(1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups:
                group["lr"] = lr*group["lr_scale"]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = wrapped(x.float())
            require_finite(logits, "v3 training logits", world > 1)
            loss = dense_loss(logits, y, check=False)
            require_finite(loss, "v3 training loss", world > 1)
            loss.backward()
            updates = dense_step(model, optimizer, ema, updates, cfg["ema_decay"], world > 1)
            total += float(loss.detach())
            if (step+1) % 40 == 0:
                print(f"Member {member} rank {rank}: epoch {epoch+1}/{cfg['dense_epochs']}, "
                      f"step {step+1}/{steps}, loss {total/(step+1):.4f}", flush=True)
        counts = np.zeros(4, np.float64)
        evaluate = not full and (epoch == 0 or (epoch+1) % cfg["pq_every"] == 0 or epoch+1 == cfg["dense_epochs"])
        if evaluate:
            counts = fixed_rule_counts(ema, manifest["splits"]["calibration"][rank::world], device, cfg)
        values = torch.tensor([total/max(steps, 1), *counts], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(values)
        values = values.cpu().numpy()
        pq = summarize(values[1:])["PQ"] if evaluate else None
        improved = evaluate and pq > best
        if improved:
            best, best_epoch = pq, epoch+1
        completed = epoch+1
        if full:
            best_epoch = completed
        if rank == 0:
            checkpoint = dict(state=ema.state_dict(), spec=spec, epoch=completed,
                fingerprint=cfg["fingerprint"], precision="float32", numerics_version=3, all_data_refit=full)
            checkpoint_save(out / "last.pt", checkpoint)
            if improved or full:
                checkpoint_save(out / "best.pt", {**checkpoint, "checkpoint_PQ": None if full else best})
            checkpoint_save(last, dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                epoch=completed, best=best, best_epoch=best_epoch, updates=updates,
                steps=steps, batch=batch, fingerprint=cfg["fingerprint"]))
            log = dict(epoch=completed, loss=float(values[0]/world), fixed_rule_calibration_PQ=pq,
                lr=lr, updates=updates, seconds=round(time.time()-started, 2),
                peak_vram_gib=round(torch.cuda.max_memory_allocated(device)/2**30, 3), batch=batch, steps=steps)
            with (out / "history.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(log, allow_nan=False)+"\n")
            print(json.dumps(log), flush=True)
        # Both ranks stop together. Leave a measured epoch margin for checkpoint I/O.
        stop = torch.tensor(int(time.time() + max(60., time.time()-started)*1.15 > stop_at), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if stop.item():
            if rank == 0:
                print("Training allocation reached; exporting the best completed checkpoint.", flush=True)
            break
    if rank == 0:
        save_json(out / "complete.json", dict(best_epoch=best_epoch, fixed_rule_calibration_PQ=None if full else best,
            epochs=completed, requested_epochs=cfg["dense_epochs"], updates=updates,
            checkpoint_sha=file_sha(out / "best.pt"), measured_plan=resource,
            training_seconds=time.time()-training_start))
        last.unlink(missing_ok=True)
    if world > 1:
        dist.barrier()


def command(config_path, task, *args, distributed=False):
    cfg = load_json(config_path)
    script = str(Path(WORKER_SCRIPT).resolve())
    cmd = [sys.executable]
    if distributed and len(cfg["gpu_ids"]) > 1:
        cmd += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc_per_node={len(cfg['gpu_ids'])}"]
    cmd += [script, "--config", str(config_path), "--task", task, *map(str, args)]
    print("Launching:", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, start_new_session=os.name != "nt")
    try:
        code = process.wait(timeout=max(1, check_deadline(cfg, task)-60))
        if code:
            raise RuntimeError(f"{task} worker failed with exit status {code}")
    except BaseException:
        terminate_worker(process)
        raise


def terminate_worker(process):
    if process.poll() is not None:
        return
    if os.name != "nt":
        import signal
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    else:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            import signal
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        else:
            process.kill()


def train_all(cfg, full=False):
    if full:
        require_report_ready(cfg)
        manifest = load_json(Path(cfg["run"])/"manifest.json")
        prepare_instance_targets(cfg, manifest, include_holdout=True)
    path = Path(cfg["run"]) / "config.json"
    members = selected_dense_ids(cfg) if full else range(len(cfg["dense_models"]))
    for member in members:
        reserve = 900 if full else min(3600, cfg["total_hours"]*900)
        check_deadline(cfg, "start next model", reserve=reserve)
        command(path, "dense", "--member", member, *(["--full"] if full else []), distributed=True)


def prepare_detector(manifest):
    # Retained hook in prepare(); v3 has no detector dependencies or artifacts.
    return None


def model_paths(cfg, full=False, all_members=False):
    root = Path(cfg["run"]) / ("full_models" if full else "selection_models")
    return dict(dense=[str(root/f"dense_{i}"/"best.pt") for i in selected_dense_ids(cfg, all_members)])


def prediction_key(cfg, paths):
    return digest(dict(runtime=VERSION, run=cfg["fingerprint"],
        checkpoints=[file_sha(p) for p in paths["dense"]],
        tile=cfg["tile_size"], overlap=cfg["tile_overlap"], tta=cfg["tta"]))[:16]


def predict_worker(config_path, part, gpu_slot=0, full=False):
    cfg = load_json(config_path)
    manifest = load_json(Path(cfg["run"]) / "manifest.json")
    device = torch.device(f"cuda:{cfg['gpu_ids'][gpu_slot]}")
    torch.cuda.set_device(device)
    paths = model_paths(cfg, full, all_members=part == "calibration")
    folder = Path(cfg["run"])/"predictions"/prediction_key(cfg, paths)/part
    folder.mkdir(parents=True, exist_ok=True)
    models = []
    for path in paths["dense"]:
        checkpoint = load_dense_checkpoint(path)
        model = build_dense(checkpoint["spec"])
        model.load_state_dict(checkpoint["state"])
        models.append(model.to(device).eval())
    variant = "avg"
    chosen = Path(cfg["run"])/"selected_pipeline.json"
    if chosen.is_file() and part != "calibration":
        variant = load_json(chosen)["recipe"]["set"].split("_", 1)[0]
    sources = manifest["splits"][part][gpu_slot::len(cfg["gpu_ids"])]
    for i, source in enumerate(sources):
        check_deadline(cfg, f"{part} prediction", reserve=90)
        target = folder/(source["stem"]+".npz")
        if target.is_file():
            with np.load(target) as cache:
                if int(cache["schema"]) != 3 or cache["prob"].shape != (2, SIZE, SIZE):
                    raise ValueError(f"Invalid v3 cache: {target}")
            continue
        image = cv2.imread(source["cache"])
        probability = np.zeros((2, SIZE, SIZE), np.float32)
        individual = []
        for model in models:
            prediction = dense_predict(model, image, device, cfg["tile_size"], cfg["tile_overlap"],
                cfg["inference_batch"], cfg["tta"])
            probability += prediction/len(models)
            if part == "calibration":
                individual.append(quantize_probabilities(prediction))
        with target.with_suffix(".tmp").open("wb") as f:
            np.savez_compressed(f, prob=quantize_probabilities(probability), schema=np.array(3),
                variant=np.array(variant), member_ids=np.array(selected_dense_ids(cfg, all_members=part=="calibration")),
                member_prob=np.stack(individual) if individual else np.empty(0, np.uint8))
        os.replace(target.with_suffix(".tmp"), target)
        print(f"GPU {device.index}: {part} {i+1}/{len(sources)} {source['stem']}", flush=True)


def predict_part(cfg, part, full=False):
    paths = model_paths(cfg, full, all_members=part=="calibration")
    folder = Path(cfg["run"])/"predictions"/prediction_key(cfg, paths)/part
    processes = []
    try:
        for slot in range(len(cfg["gpu_ids"])):
            cmd = [sys.executable, str(Path(WORKER_SCRIPT).resolve()), "--config", str(Path(cfg["run"])/"config.json"),
                "--task", "predict", "--part", part, "--gpu-slot", str(slot)]
            if full:
                cmd.append("--full")
            processes.append(subprocess.Popen(cmd, start_new_session=os.name!="nt"))
        for p in processes:
            code = p.wait(timeout=max(1, check_deadline(cfg, part)-60))
            if code:
                raise RuntimeError(f"{part} prediction worker failed: {code}")
    finally:
        for p in processes:
            terminate_worker(p)
    expected = {s["stem"]+".npz" for s in load_json(Path(cfg["run"])/"manifest.json")["splits"][part]}
    if {p.name for p in folder.glob("*.npz")} != expected:
        raise RuntimeError("Prediction cache does not cover the complete partition")
    save_json(folder/"provenance.json", dict(models=paths, physical_images=len(expected), schema=3))
    return folder


def candidate_sets(source, prediction_path, thresholds, maximum=192, member=None):
    with np.load(prediction_path) as cache:
        if int(cache["schema"]) != 3:
            raise ValueError("v3 requires newly computed probability maps")
        arrays = {str(cache["variant"]): cache["prob"].astype(np.float32)/255}
        if cache["member_prob"].size:
            arrays = {"avg": cache["prob"].astype(np.float32)/255}
            arrays.update({f"v{int(i)}": p.astype(np.float32)/255
                           for i, p in zip(cache["member_ids"], cache["member_prob"])})
    image = cv2.imread(source["cache"])
    result = {}
    for variant, (prob, edge) in arrays.items():
        for threshold in thresholds:
            for family in ("dense", "close"):
                binary = (prob >= threshold).astype(np.uint8)
                if family == "close":
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                candidates = []
                for box, mask in components(binary.astype(np.float32), .5, 20, maximum):
                    candidates.append(make_candidate(box, mask, prob, edge, image))
                result[f"{variant}_{family}_{threshold}"] = candidates
    return result


def build_candidate_cache(cfg, part, folder, refresh=False):
    manifest = load_json(Path(cfg["run"])/"manifest.json")
    profile = digest(dict(version=3, thresholds=cfg["pixel_thresholds"], maximum=cfg["max_dense_candidates"]))[:10]
    out = Path(folder)/("candidates_"+profile)
    out.mkdir(exist_ok=True)
    def process(source):
        check_deadline(cfg, f"{part} candidates", reserve=60)
        path = out/(source["stem"]+".json")
        if path.is_file() and not refresh:
            return path
        sets = candidate_sets(source, Path(folder)/(source["stem"]+".npz"),
            cfg["pixel_thresholds"], cfg["max_dense_candidates"])
        if part == "calibration":
            for candidates in sets.values():
                for c, target in zip(candidates, candidate_targets(source, candidates)):
                    c["target"] = float(target)
        save_json(path, sets)
        return path
    files = list(bounded_map(process, manifest["splits"][part], cfg["cpu_workers"]))
    print(f"{part}: cached candidates for {len(files)} physical images", flush=True)
    return out


class ComponentTree(list):
    """Nested components across probability thresholds for one model/family.

    Threshold sets (including their monotone closing) form an inclusion forest.
    A dynamic program can choose one complete parent or several disjoint children
    without cutting masks, fabricating pixels, or allowing duplicate ownership.
    """
    def __init__(self, candidates):
        seen = set()
        unique = []
        for candidate in candidates:
            key = candidate["rle"]["counts"]
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        super().__init__(unique)
        self.parents = None
        self.cached_scores = {}

    def topology(self):
        if self.parents is not None:
            return self.parents
        n = len(self)
        parents = np.full(n, -1, np.int32)
        if n:
            rles = [{**c["rle"], "counts": c["rle"]["counts"].encode()} for c in self]
            area = np.asarray([c["area"] for c in self], np.float64)
            iou = np.asarray(mu.iou(rles, rles, [0]*n), np.float64)
            intersection = iou*(area[:, None]+area[None, :])/(1+iou)
            for child in range(n):
                possible = np.flatnonzero((area>area[child]) & (intersection[:, child] >= area[child]-.01))
                if len(possible):
                    parents[child] = possible[np.argmin(area[possible])]
        self.parents = parents
        return parents


def recipe_candidates(sets, name):
    if not name.endswith("_tree"):
        return sets[name]
    memo = name+"_memo"
    if memo not in sets:
        prefix = name[:-4]  # 'v1_dense_' or 'avg_close_'
        candidates = [c for key, values in list(sets.items())
                      if key.startswith(prefix) and not key.endswith("_memo") for c in values]
        sets[memo] = ComponentTree(candidates)
    return sets[memo]


def recipes(cfg):
    variants = [f"v{i}" for i in range(len(cfg["dense_models"]))] + ["avg"]
    fixed = [dict(set=f"{variant}_{family}_{pixel}", quality=quality, cutoff=cutoff, min_area=area)
        for variant in variants for family in ("dense", "close") for pixel in cfg["pixel_thresholds"]
        for quality in (False, True) for cutoff in ([.15, .25, .35, .45] if quality else [.55, .70, .85])
        for area in (40, 100, 200)]
    trees = [dict(set=f"{variant}_{family}_tree", quality=True, cutoff=cutoff, min_area=area)
        for variant in variants for family in ("dense", "close") for cutoff in (.15, .20, .25, .30)
        for area in (40, 100, 200)]
    return fixed+trees


def select_instances(candidates, recipe, quality=None, oof=False):
    if not candidates:
        return []
    tree = isinstance(candidates, ComponentTree)
    score_key = "oof" if oof else id(quality)
    if tree and score_key in candidates.cached_scores:
        scores = candidates.cached_scores[score_key]
    elif recipe["quality"]:
        scores = (np.asarray([c["quality_oof"] for c in candidates]) if oof else
            np.clip(quality.predict(np.asarray([c["features"] for c in candidates], np.float32)), 0, 1))
        if tree:
            candidates.cached_scores[score_key] = scores
    else:
        scores = np.asarray([c["features"][1] for c in candidates])
    if tree:
        parents = candidates.topology()
        children = [[] for _ in candidates]
        for index, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(index)
        def solve(index):
            child_value, child_choice = 0., []
            for child in children[index]:
                value, choice = solve(child)
                child_value += value
                child_choice.extend(choice)
            utility = scores[index]-recipe["cutoff"] if candidates[index]["area"]>=recipe["min_area"] else -1.
            return (float(utility), [index]) if utility>max(0., child_value) else (child_value, child_choice)
        chosen = []
        for root in np.flatnonzero(parents<0):
            chosen.extend(solve(int(root))[1])
        return [candidates[i]["rle"] for i in chosen]
    # Every candidate set is the connected components of ONE map/threshold.
    # Thus masks are already disjoint. Do not allocate a 2048 canvas, re-encode,
    # or run another NMS for every one of the hundreds of calibration recipes.
    return [c["rle"] for c, score in zip(candidates, scores)
            if score >= recipe["cutoff"] and c["area"] >= recipe["min_area"]]


def paired_pq_interval(primary, alternative, seed, repeats=500):
    """Date-paired bootstrap of a calibration PQ difference, never report labels."""
    primary, alternative = np.asarray(primary), np.asarray(alternative)
    if primary.shape != alternative.shape or primary.ndim != 2 or primary.shape[1] != 4:
        raise ValueError("Expected aligned [date, four pooled metric counts] arrays")
    rng = np.random.default_rng(seed)
    difference = []
    for _ in range(repeats):
        sample = rng.integers(len(primary), size=len(primary))
        difference.append(summarize(primary[sample].sum(0))["PQ"]-
                          summarize(alternative[sample].sum(0))["PQ"])
    return list(map(float, np.quantile(difference, [.025, .975])))


def stable_recipe_choice(cfg, candidates, counts, date_counts):
    """Prefer averaging/plain components when their disadvantage is uncertain."""
    pq = np.asarray([summarize(c)["PQ"] for c in counts])
    chosen = int(pq.argmax())
    comparisons = []
    dates = np.stack(list(date_counts.values()))
    def prefer(alternative, reason):
        nonlocal chosen
        interval = paired_pq_interval(dates[:, chosen], dates[:, alternative], cfg["seed"])
        use_alternative = interval[0] <= 0
        comparisons.append(dict(numerical_winner=candidates[chosen], alternative=candidates[alternative],
            pq_difference=float(pq[chosen]-pq[alternative]), date_paired_95ci=interval,
            chose_alternative=use_alternative, reason=reason))
        if use_alternative:
            chosen = alternative
    average = [i for i, recipe in enumerate(candidates) if recipe["set"].startswith("avg_")]
    if average and not candidates[chosen]["set"].startswith("avg_"):
        prefer(max(average, key=lambda i: pq[i]), "Prefer the model average unless the single-model gain is reliable across dates")
    variant = candidates[chosen]["set"].split("_", 1)[0]
    if "_close_" in candidates[chosen]["set"] or candidates[chosen]["set"].endswith("_tree"):
        plain = [i for i, recipe in enumerate(candidates) if recipe["set"].startswith(variant+"_dense_")
                 and not recipe["set"].endswith("_tree")]
        if plain:
            prefer(max(plain, key=lambda i: pq[i]), "Prefer unchanged connected components unless grouping/morphology gains are reliable")
    return chosen, comparisons


def calibrate(cfg, candidate_folder):
    check_deadline(cfg, "calibration", reserve=900)
    model, data = fit_quality(cfg, candidate_folder)
    sources = load_json(Path(cfg["run"])/"manifest.json")["splits"]["calibration"]
    candidates = recipes(cfg)
    def score_source(source):
        check_deadline(cfg, "calibration sweep", reserve=600)
        counts, numbers = np.zeros((len(candidates), 4), np.float64), np.zeros(len(candidates), np.int64)
        sets = data[source["stem"]]
        for i, recipe in enumerate(candidates):
            pred = select_instances(recipe_candidates(sets, recipe["set"]), recipe, model, oof=True)
            numbers[i] = len(pred)
            for view in source["views"]:
                counts[i] += metric_counts(view["rles"], pred)
        return counts, numbers
    counts, numbers = np.zeros((len(candidates), 4)), np.zeros(len(candidates), np.int64)
    date_counts = defaultdict(lambda: np.zeros_like(counts))
    for source, (c, n) in zip(sources, bounded_map(score_source, sources, cfg["cpu_workers"])):
        counts += c
        numbers += n
        date_counts[source["date"]] += c
    grid = [dict(**r, **summarize(c), physical_predictions=int(n)) for r, c, n in zip(candidates, counts, numbers)]
    selected_index, comparisons = stable_recipe_choice(cfg, candidates, counts, date_counts)
    selected = grid[selected_index]
    grid.sort(key=lambda x: x["PQ"], reverse=True)
    chosen = {k: selected[k] for k in ("set", "quality", "cutoff", "min_area")}
    variant = chosen["set"].split("_", 1)[0]
    ids = list(range(len(cfg["dense_models"]))) if variant=="avg" else [int(variant[1:])]
    save_json(Path(cfg["run"])/"dense_selection.json", dict(member_ids=ids, fingerprint=cfg["fingerprint"]))
    save_json(Path(cfg["run"])/"calibration_grid.json", grid)
    save_json(Path(cfg["run"])/"calibration_stability.json", dict(comparisons=comparisons,
        bootstrap_group="calibration date", report_used=False, selected=selected))
    save_json(Path(cfg["run"])/"selected_pipeline.json", dict(recipe=chosen, calibration=selected,
        score_type="selection score; crossfitted quality; checkpoint/recipe selection is not independent CV",
        final_quality_fit="calibration dates only; report untouched"))
    print("Frozen pipeline:", json.dumps(selected), flush=True)
    return chosen


def add_report_diagnostics(cfg, candidate_folder, recipe):
    """Bootstrap dates and report misses by size without tuning on the report."""
    import joblib
    model = joblib.load(Path(cfg["run"])/"quality_model.joblib")
    sources = load_json(Path(cfg["run"])/"manifest.json")["splits"]["report"]
    dates = defaultdict(lambda: np.zeros(4, np.float64))
    areas = {"under_500": [0, 0], "500_to_1999": [0, 0], "2000_plus": [0, 0]}
    for s in sources:
        data = load_json(Path(candidate_folder)/(s["stem"]+".json"))
        pred = select_instances(recipe_candidates(data, recipe["set"]), recipe, model)
        for view in s["views"]:
            dates[s["date"]] += metric_counts(view["rles"], pred)
            gt = [{**r, "counts": r["counts"].encode()} for r in view["rles"]]
            pr = [{**r, "counts": r["counts"].encode()} for r in pred]
            hit = (np.asarray(mu.iou(gt, pr, [0]*len(pr)))>.5).any(1) if gt and pr else np.zeros(len(gt), bool)
            for r, matched in zip(gt, hit):
                area = int(mu.area(r))
                key = "under_500" if area<500 else "500_to_1999" if area<2000 else "2000_plus"
                areas[key][0] += int(matched)
                areas[key][1] += 1
    counts = np.asarray(list(dates.values()))
    rng = np.random.default_rng(cfg["seed"])
    scores = [summarize(counts[rng.integers(len(counts), size=len(counts))].sum(0))["PQ"] for _ in range(1000)]
    report = load_json(Path(cfg["run"])/"report_metrics.json")
    report.update(date_bootstrap_95ci=list(map(float, np.quantile(scores, [.025, .975]))),
        recall_by_area={k: dict(matched=m, total=n, recall=m/n if n else 0.) for k, (m, n) in areas.items()},
        total_session_seconds=time.time()-cfg["session_started"],
        target_public_pq=.5, target_achieved_on_leaderboard="unverified")
    save_json(Path(cfg["run"])/"report_metrics.json", report)
    return report


def finish_submission_run(cfg, test_candidates, recipe):
    """Keep the audited validated CSV before an optional separately timed refit."""
    run = Path(cfg["run"])
    smoke = cfg["mode"]=="smoke"
    prefix = "smoke_" if smoke else ""
    validated_cfg = {**cfg, "refit_full": False}
    original = export_submission(validated_cfg, test_candidates, recipe)
    validated = run/(prefix+"submission_validated.csv")
    shutil.copy2(original, validated)
    audit = load_json(run/"submission_audit.json")
    audit["file"] = str(validated)
    save_json(run/"validated_submission_audit.json", audit)
    variants = dict(validated=dict(file=str(validated), sha256=file_sha(validated), all_data_refit=False),
                    refit=None, default_file=str(original))
    if cfg.get("refit_full", False):
        needed = cfg["refit_hours"]*3600 + (.65*3600 if not smoke else 60)
        if deadline_remaining(cfg) < needed:
            variants["refit_skipped"] = "Insufficient measured session time; the validated CSV is preserved"
            print(variants["refit_skipped"], flush=True)
        else:
            train_all(cfg, full=True)
            full_maps = predict_part(cfg, "test", full=True)
            full_candidates = build_candidate_cache(cfg, "test", full_maps)
            original = export_submission({**cfg, "refit_full": True}, full_candidates, recipe)
            refit = run/(prefix+"submission_refit.csv")
            shutil.copy2(original, refit)
            audit = load_json(run/"submission_audit.json")
            audit["file"] = str(refit)
            save_json(run/"refit_submission_audit.json", audit)
            variants["refit"] = dict(file=str(refit), sha256=file_sha(refit), all_data_refit=True,
                report_PQ=None, note="Report score belongs to the earlier validated models")
    save_json(run/"submission_variants.json", variants)
    return original, variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", choices=["dense", "predict"], required=True)
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--part", default="test")
    parser.add_argument("--gpu-slot", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.task=="dense":
            train_dense(args.config, args.member, args.full)
        else:
            predict_worker(args.config, args.part, args.gpu_slot, args.full)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
