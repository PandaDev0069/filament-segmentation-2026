from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"
import argparse,copy,csv,gc,hashlib,json,math,random,shutil,subprocess,sys,time,types
from collections import defaultdict,deque,OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mu
from scipy import ndimage as ndi
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset,DistributedSampler
cv2.setNumThreads(1)
torch.set_num_threads(1)
SIZE=2048

FEATURES = ["log_area", "mean_probability", "max_probability", "q90_probability", "probability_std",
    "boundary_probability", "fill", "aspect", "width_max", "width_mean", "gray_inside", "local_contrast",
    "radius", "detector_confidence", "tta_support", "is_detection", "box_fill", "perimeter_ratio"]

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")
    os.replace(temp, path)

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def require_finite(tensor, context, synchronize=False):
    """All training ranks stop together BEFORE backward/step on a bad value."""
    bad = (~torch.isfinite(tensor.detach()).all()).to(dtype=torch.int32)
    if synchronize and dist.is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f"{context}: NaN/Inf detected. No invalid checkpoint or probability cache is accepted.")

def require_finite_state(value, context="checkpoint"):
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            require_finite(value, context)
    elif isinstance(value, dict):
        for key, item in value.items():
            require_finite_state(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_state(item, f"{context}[{index}]")
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise FloatingPointError(f"{context}: non-finite metadata")

def bounded_map(fn, items, workers=4):
    """At most four active/pending native images; never queue the entire dataset."""
    items = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            item = next(items, None)
            if item is not None:
                pending.append(pool.submit(fn, item))
        while pending:
            yield pending.popleft().result()
            item = next(items, None)
            if item is not None:
                pending.append(pool.submit(fn, item))

def encode(mask):
    rle = mu.encode(np.asfortranarray(mask, dtype=np.uint8))
    return dict(size=[int(x) for x in rle["size"]], counts=rle["counts"].decode("ascii"))

def decode(rle):
    rle = dict(rle)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mu.decode(rle).astype(bool)

def annotation_rle(ann, shape=(SIZE, SIZE)):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        r = mu.merge(mu.frPyObjects(seg, *shape))
    elif isinstance(seg["counts"], list):
        r = mu.frPyObjects(seg, *shape)
    else:
        r = dict(seg)
        if isinstance(r["counts"], str):
            r["counts"] = r["counts"].encode("ascii")
    return dict(size=list(shape), counts=r["counts"].decode("ascii"))

def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (SIZE, SIZE):
        raise ValueError(f"Expected a readable native {SIZE}x{SIZE} image: {path}")
    return image

def normalize_solar(gray):
    """Always normalize. Uncertain geometry disables radial correction only.

    Three uint8 channels: global standardization, bounded radial correction,
    CLAHE of the corrected view. No hard limb clipping or morphology inpainting.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("uint8 grayscale required")
    h, w = gray.shape
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    threshold, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    good, cx, cy, radius = False, w / 2, h / 2, .45 * min(h, w)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if len(contour) >= 5:
            (x, y), (a, b), _ = cv2.fitEllipse(contour)
            good = (100 < x < 156 and 100 < y < 156 and 180 < min(a, b) < 253
                    and max(a, b) / max(min(a, b), 1) < 1.13)
            if good:
                cx, cy = x * w / 256, y * h / 256
                radius = (a + b) * min(h, w) / (4 * 256)
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt(((xx - cx) / radius)**2 + ((yy - cy) / radius)**2)
    support = (rr < .88) & (gray > max(5, threshold * .35))
    if support.sum() < h * w * .1:
        support = (rr < .75)
    values = gray[support].astype(np.float32)
    mean, std = float(values.mean()), max(float(values.std()), 8.)
    raw = np.clip(128 + 32 * (gray.astype(np.float32) - mean) / std, 0, 255)
    corrected = gray.astype(np.float32)
    if good:
        # Statistics are estimated at 256² to avoid repeated full-disk sorting.
        sr = cv2.resize(rr.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA)
        bins = np.minimum((sr * 48).astype(int), 63)
        profile = np.array([np.median(small[(bins == k) & (small > 5)])
                            if np.any((bins == k) & (small > 5)) else np.nan for k in range(48)])
        valid = np.flatnonzero(np.isfinite(profile))
        if len(valid) > 24:
            profile = np.interp(np.arange(48), valid, profile[valid])
            profile = ndi.gaussian_filter1d(profile, 2)
            reference = float(np.median(profile[5:35]))
            gain = np.clip(reference / np.maximum(profile, 20), .75, 1.35)
            correction = np.interp(np.minimum(rr, 47 / 48) * 48, np.arange(48), gain)
            corrected *= correction.astype(np.float32)
    cvalues = corrected[support]
    corrected = np.clip(128 + 32 * (corrected - float(cvalues.mean())) /
                        max(float(cvalues.std()), 8.), 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(16, 16)).apply(corrected)
    return np.stack([raw.astype(np.uint8), corrected, clahe], -1), dict(
        geometry_ok=bool(good), mean=mean, std=std)

def device_setup(cfg, distributed=False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. Use Kaggle GPU T4 x2.")
    local = int(os.environ.get("LOCAL_RANK", 0))
    gpu = cfg["gpu_ids"][local]
    torch.cuda.set_device(gpu)
    if distributed and len(cfg["gpu_ids"]) > 1:
        dist.init_process_group("gloo" if os.name == "nt" else "nccl", timeout=timedelta(minutes=45))
    return torch.device(f"cuda:{gpu}"), local, len(cfg["gpu_ids"])

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def update_ema(ema, model, decay):
    with torch.no_grad():
        for dst, src in zip(ema.parameters(), model.parameters()):
            dst.lerp_(src.detach(), 1 - decay)
        for dst, src in zip(ema.buffers(), model.buffers()):
            dst.copy_(src.detach())

def metric_counts(gt, predicted):
    if gt and predicted:
        a = [{**r, "counts": r["counts"].encode()} for r in gt]
        b = [{**r, "counts": r["counts"].encode()} for r in predicted]
        iou = np.asarray(mu.iou(a, b, [0] * len(b)))
    else:
        iou = np.zeros((len(gt), len(predicted)))
    hit = iou > .5
    return np.array([iou[hit].sum(), hit.sum(), (~hit.any(0)).sum(), (~hit.any(1)).sum()], np.float64)

def summarize(counts):
    iou, tp, fp, fn = map(float, counts)
    denominator = tp + .5 * (fp + fn)
    return dict(PQ=iou / denominator if denominator else 0., SQ=iou / tp if tp else 0.,
                RQ=tp / denominator if denominator else 0., TP=int(tp), FP=int(fp), FN=int(fn), sum_IoU=iou)

def components(prob, threshold=.5, minimum=20, maximum=None):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((prob >= threshold).astype(np.uint8), 8)
    ids = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum) + 1
    if maximum is not None and len(ids) > maximum:
        sums = np.bincount(labels.ravel(), weights=prob.ravel(), minlength=n)
        means = sums[ids] / stats[ids, cv2.CC_STAT_AREA]
        ids = ids[np.argsort(-means, kind="stable")[:maximum]]
    for k in ids:
        x, y, w, h, area = map(int, stats[k])
        if area >= minimum:
            yield (x, y, x + w, y + h), labels[y:y + h, x:x + w] == k

def full_rle(box, mask, shape=None):
    shape = (SIZE, SIZE) if shape is None else shape
    x0, y0, x1, y1 = box
    if mask.shape != (y1 - y0, x1 - x0):
        raise ValueError("ROI and mask shapes differ")
    if mask.size > shape[0] * shape[1] // 8:
        image = np.zeros(shape, np.uint8)
        image[y0:y1, x0:x1] = mask
        return encode(image)
    # Encode sparse ROIs directly in the GLOBAL column-major index space. Avoid
    # allocating/encoding a 4-million-pixel zero canvas for every tiny candidate.
    cols, rows = np.nonzero(mask.T)
    indices = (cols.astype(np.int64) + x0) * shape[0] + rows + y0
    if not len(indices):
        counts = [shape[0] * shape[1]]
    else:
        breaks = np.flatnonzero(np.diff(indices) != 1)
        starts = indices[np.r_[0, breaks + 1]]
        ends = indices[np.r_[breaks, len(indices) - 1]]
        runs = np.empty(len(starts) * 2, np.int64)
        runs[0::2] = starts - np.r_[-1, ends[:-1]] - 1
        runs[1::2] = ends - starts + 1
        counts = runs.tolist()
        tail = shape[0] * shape[1] - int(ends[-1]) - 1
        if tail:
            counts.append(tail)
    rle = mu.frPyObjects(dict(size=list(shape), counts=counts), *shape)
    return dict(size=list(shape), counts=rle["counts"].decode("ascii"))

def checkpoint_save(path, value):
    require_finite_state(value, str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)

def mask_features(box, mask, prob, edge, image, confidence=0., support=0., detection=False):
    x0, y0, x1, y1 = box
    p = prob[y0:y1, x0:x1][mask]
    e = edge[y0:y1, x0:x1][mask]
    gray = image[y0:y1, x0:x1, 1].astype(np.float32) / 255
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1][mask]
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~mask
    contrast = float(gray[ring].mean() - gray[mask].mean()) if ring.any() else 0.
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    points = np.argwhere(mask)
    height, width = mask.shape
    area = int(mask.sum())
    return [math.log1p(area), float(p.mean()), float(p.max()), float(np.quantile(p, .9)), float(p.std()),
        float(e.mean()), area / (height * width), max(height, width) / max(min(height, width), 1),
        float(distance.max()), float(distance.mean()), float(gray[mask].mean()), contrast,
        float(np.hypot(points[:, 1].mean() + x0 - SIZE/2, points[:, 0].mean() + y0 - SIZE/2) / (SIZE/2)),
        float(confidence), float(support), float(detection), area / (height * width),
        float(perimeter / math.sqrt(max(area, 1)))]

def make_candidate(box, mask, prob, edge, image, confidence=0., support=0., detection=False):
    return dict(rle=full_rle(box, mask), local_rle=encode(mask), box=list(map(int, box)), area=int(mask.sum()),
        features=mask_features(box, mask, prob, edge, image, confidence, support, detection))

def candidate_targets(source, candidates):
    """Expected credited IoU over independent annotator records, using official >0.5."""
    if not candidates:
        return np.zeros(0, np.float32)
    pred = [{**c["rle"], "counts": c["rle"]["counts"].encode()} for c in candidates]
    result = np.zeros(len(pred), np.float64)
    for view in source["views"]:
        gt = [{**r, "counts": r["counts"].encode()} for r in view["rles"]]
        if gt:
            iou = np.asarray(mu.iou(gt, pred, [0] * len(pred)))
            result += (iou * (iou > .5)).sum(0)
    return (result / max(len(source["views"]), 1)).astype(np.float32)

def quality_estimator(seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=140, max_leaf_nodes=7, learning_rate=.05,
        min_samples_leaf=30, l2_regularization=2., early_stopping=False, random_state=seed)
b=types.SimpleNamespace(**{name:globals()[name] for name in ['FEATURES', 'annotation_rle', 'bounded_map', 'candidate_targets', 'checkpoint_save', 'components', 'decode', 'device_setup', 'digest', 'encode', 'file_sha', 'full_rle', 'load_json', 'make_candidate', 'mask_features', 'metric_counts', 'normalize_solar', 'quality_estimator', 'read_gray', 'require_finite', 'require_finite_state', 'save_json', 'seed_all', 'summarize', 'update_ema']})

EXTRA_FEATURES=["crop_quality","crop_probability","seed_coverage","is_refined"]
def sam_remaining(cfg, reserve=0):
    seconds = cfg["session_started"]+cfg["total_hours"]*3600-time.time()
    if seconds < reserve:
        raise TimeoutError(f"SAM refinement has {seconds/60:.1f} minutes remaining; preserving artifacts")
    return seconds

def square_roi(box, cfg):
    x0, y0, x1, y1 = map(float, box)
    side = int(math.ceil(max(x1-x0, y1-y0)*cfg["roi_context"]/16)*16)
    side = min(2048, cfg["maximum_roi"], max(cfg["minimum_roi"], side))
    left = int(np.clip(round((x0+x1-side)/2), 0, 2048-side))
    top = int(np.clip(round((y0+y1-side)/2), 0, 2048-side))
    return [left, top, left+side, top+side]

def discover_prior(cfg):
    matches = []
    for path in Path(cfg["input_root"]).rglob("manifest.json"):
        try:
            manifest = b.load_json(path)
        except (OSError, ValueError):
            continue
        if manifest.get("config", {}).get("fingerprint")==cfg["prior_fingerprint"]:
            matches.append((path, manifest))
    if len(matches)!=1:
        raise ValueError("Attach output of pandadev0069/filament-v3-warm-start-pilot, containing the 90b4b0b58b run")
    return matches[0]

def prior_crossfit_scores(cfg,prior_root,prior):
    """Reproduce prior date-OOF quality scores for HONEST calibration proposals.

    The old final regressor saw all calibration dates. Using its fitted scores
    to prefilter new calibration proposals would leak their labels into selection.
    Report/test continue to use that final regressor, as in the original pipeline.
    """
    recipe=b.load_json(prior_root/"selected_pipeline.json")["recipe"]
    if not recipe["quality"]:return {}
    folders=list((prior_root/"predictions").glob("*/calibration/candidates_*"))
    folder=next((p for p in folders if all((p/(s["stem"]+".json")).is_file() for s in prior["splits"]["calibration"])),None)
    if folder is None:raise ValueError("The full prior calibration candidate cache is needed for date-OOF reconstruction")
    x=[];y=[];dates=[];indices=[];result={}
    for source in prior["splits"]["calibration"]:
        sets=b.load_json(folder/(source["stem"]+".json"))
        result[source["stem"]]=[0.]*len(sets[recipe["set"]])
        for name,values in sets.items():
            for j,c in enumerate(values):
                x.append(c["features"]);y.append(c["target"]);dates.append(source["date"])
                indices.append((source["stem"],name,j))
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);dates=np.asarray(dates)
    seed=prior["config"]["seed"]
    unique=np.array(sorted(set(dates)));np.random.default_rng(seed).shuffle(unique)
    folds=min(prior["config"]["quality_folds"],len(unique));assignment={date:i%folds for i,date in enumerate(unique)}
    scores=np.zeros(len(y),np.float32)
    for fold in range(folds):
        valid=np.asarray([assignment[d]==fold for d in dates]);model=b.quality_estimator(seed+fold)
        model.fit(x[~valid],y[~valid]);scores[valid]=np.clip(model.predict(x[valid]),0,1)
    for (stem,name,j),score in zip(indices,scores):
        if name==recipe["set"]:result[stem][j]=float(score)
    b.save_json(Path(cfg["run"])/"prior_quality_oof.json",dict(scores=result,seed=seed,folds=folds,
        group="prior calibration date",fitted_prior_quality_not_used_for_calibration_proposal_filter=True))
    return result

def grouped_records(source, part, cfg):
    views = source["views"]
    nodes = [(vi, r, int(mu.area({**r,"counts":r["counts"].encode()})))
             for vi,view in enumerate(views) for r in view["rles"]]
    groups = []
    for vi,rle,area in sorted(nodes,key=lambda x:-x[2]):
        available = [i for i,g in enumerate(groups) if g["rles"][vi] is None]
        match, score = -1, 0.
        if available:
            a = [{**rle,"counts":rle["counts"].encode()}]
            refs = [{**groups[i]["reference"],"counts":groups[i]["reference"]["counts"].encode()} for i in available]
            overlap = np.asarray(mu.iou(a,refs,[0]*len(refs)))[0]
            j = int(overlap.argmax())
            if overlap[j]>=.25:
                match, score = available[j], overlap[j]
        if match<0:
            groups.append(dict(rles=[None]*len(views), reference=rle))
            match = len(groups)-1
        groups[match]["rles"][vi] = rle
    records = []
    for group in groups:
        boxes = [mu.toBbox({**r,"counts":r["counts"].encode()}) for r in group["rles"] if r]
        boxes = np.asarray(boxes)
        bounds = [boxes[:,0].min(),boxes[:,1].min(),(boxes[:,0]+boxes[:,2]).max(),(boxes[:,1]+boxes[:,3]).max()]
        roi=square_roi(bounds,cfg)
        rotation=0;flip=False
        if part=="train":
            rng=np.random.default_rng(cfg["seed"]+int(b.digest([source["stem"],len(records)])[:8],16))
            variant={**cfg,"roi_context":float(rng.uniform(1.4,2.0))}
            roi=square_roi(bounds,variant);side=roi[2]-roi[0]
            for axis in (0,1):
                lower=max(0,int(math.ceil(bounds[axis+2]-side)))
                upper=min(2048-side,int(math.floor(bounds[axis])))
                if lower<=upper:
                    roi[axis]=int(np.clip(round(roi[axis]+rng.uniform(-.12,.12)*side),lower,upper))
                    roi[axis+2]=roi[axis]+side
            rotation=int(rng.integers(4));flip=bool(rng.random()<.5)
        records.append(dict(stem=source["stem"], part=part, roi=roi,rotation=rotation,flip=flip,
            rles=group["rles"], nviews=len(views), positive=True, image=source["cache"]))
    # Dark, on-disk negatives with no annotation in ANY available annotator view.
    image = cv2.imread(source["cache"])
    union = np.zeros((2048,2048),bool)
    for _,r,_ in nodes:
        union |= b.decode(r)
    rng = np.random.default_rng(cfg["seed"]+int(b.digest(source["stem"])[:8],16))
    points = rng.integers(200,1848,size=(160,2))
    points = points[((points-1024)**2).sum(1)<820**2]
    intensity = image[points[:,0],points[:,1],1]
    negative_boxes = []
    for j in np.argsort(intensity):
        y,x = points[j]
        if intensity[j]<10:
            continue
        roi = square_roi([x-32,y-32,x+32,y+32],cfg)
        x0,y0,x1,y1 = roi
        if union[y0:y1,x0:x1].any():
            continue
        if any(abs(x0-a[0])+abs(y0-a[1])<150 for a in negative_boxes):
            continue
        negative_boxes.append(roi)
        records.append(dict(stem=source["stem"],part=part,roi=roi,rles=[None]*len(views),
            nviews=len(views),positive=False,image=source["cache"],
            rotation=int(rng.integers(4)) if part=="train" else 0,
            flip=bool(rng.random()<.5) if part=="train" else False))
        if len(negative_boxes)==2:
            break
    return records

def crop_transform(array,record):
    transformed=np.rot90(array,int(record.get("rotation",0)),axes=(-2,-1))
    if record.get("flip",False):transformed=transformed[...,::-1]
    return np.ascontiguousarray(transformed)

def soft_skeleton(mask,iterations=3):
    def erode(x):
        return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
    def opened(x):return F.max_pool2d(erode(x),3,1,1)
    result=F.relu(mask-opened(mask))
    for _ in range(iterations):
        mask=erode(mask);delta=F.relu(mask-opened(mask));result=result+F.relu(delta-result*delta)
    return result

def stop_sam_process(process):
    if process.poll() is not None:return
    if os.name=="nt":
        subprocess.run(["taskkill","/PID",str(process.pid),"/T","/F"],capture_output=True)
    else:
        import signal
        os.killpg(os.getpgid(process.pid),signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name=="nt":process.kill()
        else:
            import signal
            os.killpg(os.getpgid(process.pid),signal.SIGKILL)

def await_sam(cfg,processes):
    try:
        for process in processes:
            code=process.wait(timeout=max(1,sam_remaining(cfg,30)))
            if code:raise RuntimeError(f"SAM worker failed with exit status {code}")
    finally:
        for process in processes:stop_sam_process(process)

def prior_quality(cfg):
    return load_quality(Path(cfg["prior_root"])/"quality_model.joblib")

def load_quality(path):
    # Some sklearn/Cython wheels serialize their extension under the short name
    # '_loss'. Resolve that name to the SAME installed sklearn extension; do not
    # change the estimator or refit it during inference.
    import importlib
    import joblib
    sys.modules.setdefault("_loss",importlib.import_module("sklearn._loss._loss"))
    return joblib.load(path)

def prior_instances(source,manifest,quality):
    recipe=manifest["prior_calibration"]["recipe"]
    candidates=b.load_json(source["prior_candidates"])[recipe["set"]]
    if recipe["quality"]:
        scores=(source["prior_oof"] if "prior_oof" in source else
                np.clip(quality.predict(np.asarray([c["features"] for c in candidates],np.float32)),0,1) if candidates else [])
    else:scores=[c["features"][1] for c in candidates]
    if len(scores)!=len(candidates):raise ValueError("Prior OOF candidate alignment mismatch")
    selected=[copy.deepcopy(c) for c,s in zip(candidates,scores) if s>=recipe["cutoff"] and c["area"]>=recipe["min_area"]]
    return selected

def propose_seeds(source,cfg,manifest,quality,probability,image):
    base=prior_instances(source,manifest,quality)
    candidates=[]
    for c in base:c["priority"]=2.;candidates.append(c)
    for threshold in cfg["proposal_thresholds"]:
        for box,mask in b.components(probability[0],threshold,cfg["minimum_seed_area"],192):
            if max(box[2]-box[0],box[3]-box[1])>1024:continue
            c=b.make_candidate(box,mask,probability[0],probability[1],image)
            c["priority"]=float(c["features"][1])
            candidates.append(c)
    chosen=[];encoded=[]
    for c in sorted(candidates,key=lambda x:-x["priority"]):
        current={**c["rle"],"counts":c["rle"]["counts"].encode()}
        if encoded and np.asarray(mu.iou([current],encoded,[0]*len(encoded))).max()>.80:continue
        chosen.append(c);encoded.append(current)
        if len(chosen)>=cfg["max_proposals"]:break
    return base,chosen

def put_seed_in_roi(candidate,roi,probability):
    x0,y0,x1,y1=roi;a,c,d,e=candidate["box"]
    result=np.zeros((y1-y0,x1-x0),np.float32)
    left,top,right,bottom=max(x0,a),max(y0,c),min(x1,d),min(y1,e)
    if right>left and bottom>top:
        local=b.decode(candidate["local_rle"])
        result[top-y0:bottom-y0,left-x0:right-x0]=local[top-c:bottom-c,left-a:right-a]*probability[top:bottom,left:right]
    return result

def fit_refiner_quality(cfg,folder):
    import joblib
    manifest=b.load_json(Path(cfg["run"])/"manifest.json")
    data={};x=[];y=[];dates=[];index=[]
    for source in manifest["splits"]["calibration"]:
        sets=b.load_json(Path(folder)/(source["stem"]+".json"));data[source["stem"]]=sets
        for name,candidates in sets.items():
            for j,c in enumerate(candidates):
                x.append(c["features"]);y.append(c["target"]);dates.append(source["date"]);index.append((source["stem"],name,j))
    if not x:raise RuntimeError("No SAM calibration candidates")
    x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);dates=np.asarray(dates)
    unique=np.array(sorted(set(dates)));np.random.default_rng(cfg["seed"]).shuffle(unique)
    folds=min(cfg["quality_folds"],len(unique))
    if folds<2:raise ValueError("Quality crossfitting requires two calibration dates")
    assignment={date:i%folds for i,date in enumerate(unique)};oof=np.zeros(len(y),np.float32)
    for fold in range(folds):
        valid=np.asarray([assignment[d]==fold for d in dates])
        model=b.quality_estimator(cfg["seed"]+fold)
        model.fit(x[~valid],y[~valid]);oof[valid]=np.clip(model.predict(x[valid]),0,1)
    for (stem,name,j),score in zip(index,oof):data[stem][name][j]["quality_oof"]=float(score)
    model=b.quality_estimator(cfg["seed"]);model.fit(x,y)
    joblib.dump(model,Path(cfg["run"])/"refiner_quality.joblib")
    b.save_json(Path(cfg["run"])/"quality_audit.json",dict(features=list(b.FEATURES)+EXTRA_FEATURES,
        folds=folds,group="calibration date",rows=len(x),dates=len(unique),report_used=False,
        target="Expected credited IoU over original annotation views",crossfit_mae=float(np.abs(oof-y).mean())))
    return model,data

def refined_family(sets,recipe):
    if recipe["family"]=="prior":return sets["baseline"]
    refined=sets[f"refined_{recipe['threshold']}"]
    return refined+sets["baseline"] if recipe["family"]=="fusion" else refined

def select_refiner_instances(sets,recipe,quality=None,oof=False):
    candidates=refined_family(sets,recipe)
    if recipe["family"]=="prior":return [c["rle"] for c in candidates]
    if not candidates:return []
    scores=np.asarray([c["quality_oof"] for c in candidates]) if oof else np.clip(
        quality.predict(np.asarray([c["features"] for c in candidates],np.float32)),0,1)
    owned=np.zeros((2048,2048),bool);accepted=[];rles=[];areas=[]
    for i in np.argsort(-scores,kind="stable"):
        c=candidates[i]
        if scores[i]<recipe["cutoff"] or c["area"]<recipe["min_area"]:continue
        encoded={**c["rle"],"counts":c["rle"]["counts"].encode()}
        if rles:
            overlap=np.asarray(mu.iou([encoded],rles,[0]*len(rles)))[0]
            intersection=overlap*(c["area"]+np.asarray(areas))/(1+overlap)
            if overlap.max()>.5 or (intersection/np.minimum(c["area"],areas)).max()>.70:continue
        x0,y0,x1,y1=c["box"]
        local=c.get("_decoded")
        if local is None:local=b.decode(c["local_rle"])
        mask=local & ~owned[y0:y1,x0:x1]
        if int(mask.sum())<recipe["min_area"]:continue
        owned[y0:y1,x0:x1]|=mask
        full=b.full_rle(c["box"],mask)
        accepted.append(full);rles.append({**full,"counts":full["counts"].encode()});areas.append(int(mask.sum()))
    return accepted

def export_refiner(cfg,folder,recipe):
    run=Path(cfg["run"]);manifest=b.load_json(run/"manifest.json")
    model=load_quality(run/"refiner_quality.joblib")
    if cfg["mode"]=="full" and len(manifest["splits"]["test"])!=180:raise ValueError("Need all 180 test images")
    prefix="smoke_" if cfg["mode"]=="smoke" else ""
    files={"selected":run/(prefix+"submission.csv"),"baseline":run/(prefix+"submission_baseline.csv")}
    audits={}
    for kind,path in files.items():
        identifiers=set();per_image=[]
        with path.with_suffix(".tmp").open("w",newline="",encoding="utf8") as stream:
            writer=csv.writer(stream);writer.writerow(["filament_id","segmentation_rle"])
            for source in manifest["splits"]["test"]:
                sam_remaining(cfg,30)
                sets=b.load_json(Path(folder)/(source["stem"]+".json"))
                predictions=select_refiner_instances(sets,recipe if kind=="selected" else dict(family="prior"),model)
                owned=np.zeros((2048,2048),bool)
                for j,rle in enumerate(predictions,1):
                    mask=b.decode(rle)
                    if mask.shape!=(2048,2048) or not mask.any() or (owned&mask).any():raise ValueError("Invalid/overlapping output mask")
                    if b.encode(mask)["counts"]!=rle["counts"]:raise ValueError("RLE round trip failed")
                    owned|=mask;identity=f"{source['stem']}_{j}"
                    if identity in identifiers:raise ValueError("Duplicate output ID")
                    identifiers.add(identity);writer.writerow([identity,rle["counts"]])
                per_image.append(dict(image=source["name"],predictions=len(predictions)))
        os.replace(path.with_suffix(".tmp"),path)
        audits[kind]=dict(file=str(path),sha256=b.file_sha(path),masks=len(identifiers),images=len(per_image),
            per_image=per_image,roundtrip=True,nonoverlap=True,native_shape=[2048,2048])
    b.save_json(run/"submission_audits.json",audits)
    if cfg["mode"]=="full":shutil.copy2(files["selected"],Path(cfg["output_root"])/"submission.csv")
    b.save_json(run/"session_timing.json",dict(seconds=time.time()-cfg["session_started"],
        budget_seconds=cfg["total_hours"]*3600,prior_full_run_seconds=14816.277888,
        decoder_only_training=True,official_public_score="not measured"))
    print("Exported:",files,"| mask counts:",{k:v["masks"] for k,v in audits.items()},flush=True)
    return files
s=types.SimpleNamespace(**{name:globals()[name] for name in ['await_sam', 'crop_transform', 'discover_prior', 'export_refiner', 'fit_refiner_quality', 'grouped_records', 'load_quality', 'prior_crossfit_scores', 'prior_instances', 'prior_quality', 'propose_seeds', 'put_seed_in_roi', 'refined_family', 'sam_remaining', 'select_refiner_instances', 'soft_skeleton', 'square_roi', 'stop_sam_process']})

"""Image-trained instance crops with noisy box prompts and honest calibration."""

import copy
import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mu
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


CROP_WORKER = Path("/kaggle/working/box_crop_v5_worker.py")


def crop_config():
    return dict(mode="full",input_root="/kaggle/input",output_root="/kaggle/working",
        gpu_ids=[0,1],cpu_workers=4,seed=20261109,session_started=time.time(),
        model_version="box-crop-v5", total_hours=8., train_hours=5.5,
        epochs=120, steps=200, crop_size=512, validation_every=5,
        validation_records=256, max_proposals=48, proposal_thresholds=[.15, .35],
        mask_thresholds=[.35, .5, .65], inference_batch=8, tta=4,
        models=[dict(encoder="resnet34", batch=16),
                dict(encoder="tu-tf_efficientnetv2_s", batch=8)],
        lr=2e-4, encoder_lr=4e-5, ema_decay=.995,
        weight_decay=.01, minimum_calibration_gain=.01,quality_folds=5,
        minimum_seed_area=40,roi_context=1.7,minimum_roi=192,maximum_roi=1536,smoke_sources=3,
        prior_fingerprint="90b4b0b58b0caf5353dc319ca37d87d92998e19010a8cf5a2325cbbc38f31668")


def prepare_crop(cfg):
    import sklearn
    cfg = copy.deepcopy(cfg)
    if cfg["mode"] == "smoke":
        cfg.update(epochs=1, steps=2, tta=1, max_proposals=2,
                   mask_thresholds=[.5], validation_records=8, validation_every=1)
        cfg["models"] = [{**spec,"batch":2} for spec in cfg["models"]]
    elif cfg["mode"] != "full":
        raise ValueError("MODE must be full or smoke")
    path, prior = s.discover_prior(cfg)
    if sklearn.__version__ != prior["packages"]["scikit-learn"]:
        raise RuntimeError("Use the exact source quality model's scikit-learn version")
    root = Path(cfg["input_root"])
    annotations = list(root.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(annotations) != 1:
        raise ValueError("Attach exactly one official competition data tree")
    annotation = annotations[0]
    data = annotation.parent.parent
    if b.file_sha(annotation) != prior["config"]["annotation_sha"]:
        raise ValueError("Annotation checksum differs from the frozen source")
    cfg.update(prior_root=str(path.parent), annotation_sha=b.file_sha(annotation),
               fingerprint=b.digest(dict(version=cfg["model_version"],
                   config={k:v for k,v in cfg.items() if k!="session_started"},
                   worker=b.file_sha(CROP_WORKER))))
    run = Path(cfg["output_root"])/("filament_box_crop_v5_"+cfg["fingerprint"][:10])
    run.mkdir(parents=True, exist_ok=True)
    cfg["run"] = str(run)
    oof = s.prior_crossfit_scores(cfg, path.parent, prior)
    manifest = dict(config=cfg, splits=copy.deepcopy(prior["splits"]),
        prior_calibration=b.load_json(path.parent/"selected_pipeline.json"),
        prior_report=b.load_json(path.parent/"report_metrics.json"))
    checked = []
    for part, sources in list(manifest["splits"].items()):
        if cfg["mode"] == "smoke":
            sources = sources[:cfg["smoke_sources"]]
            manifest["splits"][part] = sources
        for source in sources:
            source["image"] = str(data/("test/test_images" if part=="test" else "train/train_images")/source["name"])
            source["cache"] = str(path.parent/"cache"/(source["stem"]+".png"))
            if not Path(source["cache"]).is_file():
                raise ValueError("Attach the complete source output, including its normalized image cache")
            checked.append(source)
        if part in ("calibration", "report", "test"):
            folders = list((path.parent/"predictions").glob("*/"+part))
            folder = next((p for p in folders if all((p/(src["stem"]+".npz")).is_file() for src in sources)),None)
            if folder is None:
                raise ValueError("Incomplete source probability maps")
            for source in sources:
                source["prior_map"] = str(folder/(source["stem"]+".npz"))
                candidates = list(folder.glob("candidates_*/"+source["stem"]+".json"))
                if len(candidates) != 1:
                    raise ValueError("Need the exact prior candidate cache")
                source["prior_candidates"] = str(candidates[0])
                if part == "calibration":
                    source["prior_oof"] = oof[source["stem"]]
    for source, actual in zip(checked, b.bounded_map(lambda src:b.file_sha(src["image"]),checked,4)):
        if actual != source["image_sha"]:
            raise ValueError("Raw image checksum differs from source")
    train_dates = {src["date"] for src in manifest["splits"]["train"]}
    heldout = {src["date"] for part in ("calibration","report") for src in manifest["splits"][part]}
    if train_dates & heldout:
        raise ValueError("Train and held-out dates overlap")
    groups = list(b.bounded_map(lambda pair:s.grouped_records(pair[0],pair[1],cfg),
                  [(src,"train") for src in manifest["splits"]["train"]]+
                  [(src,"validation") for src in manifest["splits"]["calibration"]],4))
    train = [record for group in groups for record in group if record["part"]=="train"]
    valid = [record for group in groups for record in group if record["part"]=="validation"]
    rng = np.random.default_rng(cfg["seed"]);rng.shuffle(train);rng.shuffle(valid)
    if cfg["mode"]=="smoke":train=train[:8]
    for record in train+valid:
        boxes = np.asarray([mu.toBbox({**r,"counts":r["counts"].encode()}) for r in record["rles"] if r])
        if len(boxes):
            record["bounds"] = [float(boxes[:,0].min()),float(boxes[:,1].min()),
                float((boxes[:,0]+boxes[:,2]).max()),float((boxes[:,1]+boxes[:,3]).max())]
        record["rotation"],record["flip"] = 0,False
    b.save_json(run/"records.json",dict(train=train,validation=valid[:cfg["validation_records"]]))
    b.save_json(run/"config.json",cfg);b.save_json(run/"manifest.json",manifest)
    b.save_json(run/"input_audit.json",dict(raw_hashes_verified=len(checked),
        train_heldout_dates_disjoint=True,ground_truth_shaped_input_masks=False,
        train_records=len(train),validation_records=min(len(valid),cfg["validation_records"]),
        annotation_sha=cfg["annotation_sha"],pretrained_models="Generic ImageNet only"))
    print("Prepared box-crop training:",run,"| train",len(train),"validation",min(len(valid),cfg["validation_records"]),flush=True)
    return cfg,manifest


def box_input(gray, bounds, roi, size):
    """Image channels and a filled rectangle; never a ground-truth-shaped seed."""
    x0,y0,x1,y1 = roi
    gray = cv2.resize(gray[y0:y1,x0:x1],(size,size),interpolation=cv2.INTER_LINEAR)
    contrast = cv2.createCLAHE(2.,(8,8)).apply(gray)
    hint = np.zeros((size,size),np.float32)
    left,top,right,bottom = map(float,bounds)
    a,c = np.floor([(left-x0)*size/(x1-x0),(top-y0)*size/(y1-y0)]).astype(int)
    d,e = np.ceil([(right-x0)*size/(x1-x0),(bottom-y0)*size/(y1-y0)]).astype(int)
    a,c,d,e = np.clip([a,c,d,e],0,size)
    hint[c:e,a:d] = 1.
    image = np.stack([gray.astype(np.float32)/255,contrast.astype(np.float32)/255,hint])
    return (image-np.array([.485,.456,.406],np.float32)[:,None,None])/np.array([.229,.224,.225],np.float32)[:,None,None]


class BoxCropDataset(Dataset):
    def __init__(self,records,cfg,augment):
        self.records,self.cfg,self.augment = records,cfg,augment
        self.epoch,self.cache = 0,OrderedDict()
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        record = self.records[index]
        rng = np.random.default_rng(self.cfg["seed"]+index+self.epoch*10_000_019)
        if record["image"] not in self.cache:
            if len(self.cache)>=12:self.cache.popitem(last=False)
            self.cache[record["image"]] = cv2.imread(record["image"],cv2.IMREAD_UNCHANGED)[:,:,0]
        gray = self.cache[record["image"]]
        roi = record["roi"]
        bounds = record.get("bounds",[roi[0]+48,roi[1]+48,roi[2]-48,roi[3]-48])
        if self.augment and record["positive"]:
            roi = s.square_roi(bounds,{**self.cfg,"roi_context":float(rng.uniform(1.4,2.4))})
        hint = np.asarray(bounds,np.float64).copy()
        if self.augment:
            cx,cy = (hint[:2]+hint[2:])/2
            w,h = np.maximum(hint[2:]-hint[:2],8)
            cx+=rng.uniform(-.15,.15)*w;cy+=rng.uniform(-.15,.15)*h
            w*=rng.uniform(.7,1.6);h*=rng.uniform(.7,1.6)
            hint=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        size = self.cfg["crop_size"]
        image = box_input(gray,hint,roi,size)
        views = np.zeros((3,size,size),np.float32)
        x0,y0,x1,y1 = roi
        for j,rle in enumerate(record["rles"]):
            if rle:
                local = b.decode(rle)[y0:y1,x0:x1].astype(np.float32)
                views[j] = cv2.resize(local,(size,size),interpolation=cv2.INTER_AREA if x1-x0>size else cv2.INTER_LINEAR)
        present = np.flatnonzero(views.sum((1,2))>0)
        target = views[int(rng.choice(present))] if len(present) else np.zeros((size,size),np.float32)
        if self.augment:
            # Photometric changes only affect real image channels, not the box.
            image[:2] = image[:2]*rng.uniform(.85,1.15)+rng.uniform(-.12,.12)
            transform=dict(rotation=int(rng.integers(4)),flip=bool(rng.random()<.5))
            image=s.crop_transform(image,transform);views=s.crop_transform(views,transform)
            target=s.crop_transform(target,transform)
        return dict(image=torch.from_numpy(np.ascontiguousarray(image)),
            target=torch.from_numpy(target[None].copy()),views=torch.from_numpy(views.copy()),
            nviews=torch.tensor(record["nviews"]))


class BoxCropUNet(nn.Module):
    def __init__(self,spec,pretrained=False):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.core=smp.Unet(encoder_name=spec["encoder"],encoder_weights="imagenet" if pretrained else None,
            in_channels=3,classes=1,decoder_channels=(128,64,32,16,16),decoder_attention_type="scse",
            aux_params=dict(pooling="avg",dropout=.1,activation=None,classes=1))
        def replace(module):
            for name,child in list(module.named_children()):
                if isinstance(child,nn.BatchNorm2d):setattr(module,name,nn.GroupNorm(math.gcd(8,child.num_features),child.num_features))
                else:replace(child)
        replace(self.core.decoder)
        self.core.decoder.blocks[-1].attention1.requires_grad_(False)
    def forward(self,image):
        mask,quality=self.core(image)
        return mask,quality[:,0]
    def train(self,mode=True):
        super().train(mode)
        for module in self.core.encoder.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self


def crop_loss(logits,quality,batch,device):
    target=batch["target"].to(device).float()
    p=logits.float().sigmoid();axes=(1,2,3)
    bce=F.binary_cross_entropy_with_logits(logits.float(),target)
    dice=(1-(2*(p*target).sum(axes)+1)/(p.sum(axes)+target.sum(axes)+1)).mean()
    line=s.soft_skeleton(p);truth=s.soft_skeleton(target)
    precision=((line*target).sum(axes)+1)/(line.sum(axes)+1)
    recall=((truth*p).sum(axes)+1)/(truth.sum(axes)+1)
    continuity=(1-2*precision*recall/(precision+recall+1e-6)).mean()
    with torch.no_grad():
        predicted=(p[:,0]>.5).float()[:,None]
        views=batch["views"].to(device).float()
        overlap=(predicted*views).sum((-2,-1))/(predicted.sum((-2,-1))+views.sum((-2,-1))-(predicted*views).sum((-2,-1))).clamp_min(1)
        active=torch.arange(3,device=device)[None]<batch["nviews"].to(device)[:,None]
        credit=(overlap*(overlap>.5)*active).sum(1)/batch["nviews"].to(device).clamp_min(1)
    return .45*bce+.45*dice+.05*continuity+.05*F.smooth_l1_loss(quality.float(),credit,beta=.1)


def probe_crop_batch(model,device,cfg,maximum,world):
    chosen,seconds=None,None
    for batch in range(maximum,0,-1):
        image=target=views=logits=quality=loss=fake=None
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            image=torch.zeros((batch,3,cfg["crop_size"],cfg["crop_size"]),device=device)
            target=torch.zeros_like(image[:,:1]);target[:,:,40:48,30:90]=1
            views=target.repeat(1,3,1,1)
            fake=dict(target=target,views=views,nviews=torch.full((batch,),3,device=device))
            model.train();timings=[]
            for _ in range(2):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize(device);before=time.time()
                logits,quality=model(image);loss=crop_loss(logits,quality,fake,device);loss.backward()
                torch.cuda.synchronize(device);timings.append(time.time()-before)
            if torch.cuda.max_memory_reserved(device)>.70*torch.cuda.get_device_properties(device).total_memory and batch>1:
                continue
            chosen,seconds=batch,timings[-1]
            break
        except torch.cuda.OutOfMemoryError:
            if batch==1:raise
        finally:
            model.zero_grad(set_to_none=True)
            del image,target,views,logits,quality,loss,fake
            gc.collect();torch.cuda.empty_cache()
    if chosen is None:raise RuntimeError("No crop batch fits with optimizer/DDP/EMA headroom")
    values=torch.tensor([float(chosen),float(seconds)],device=device,dtype=torch.float64)
    if world>1:
        dist.all_reduce(values[:1],op=dist.ReduceOp.MIN);dist.all_reduce(values[1:],op=dist.ReduceOp.MAX)
    return int(values[0].item()),float(values[1].item())


def train_crop_worker(config_path,member):
    cfg=b.load_json(config_path);run=Path(cfg["run"]);out=run/f"model_{member}"
    device,rank,world=b.device_setup(cfg,distributed=True)
    spec=cfg["models"][member]
    b.seed_all(cfg["seed"]+member*100_003)
    model=BoxCropUNet(spec,False).to(device)
    initial=torch.load(out/"initial.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(initial);del initial
    batch_size,measured=probe_crop_batch(model,device,cfg,spec["batch"],world)
    records=b.load_json(run/"records.json")
    training=BoxCropDataset(records["train"],{**cfg,"seed":cfg["seed"]+member*100_003},True)
    validation=BoxCropDataset(records["validation"][rank::world],cfg,False)
    sampler=DistributedSampler(training,world,rank,shuffle=True,seed=cfg["seed"]+member)
    workers=cfg["cpu_workers"]//world
    loader=DataLoader(training,batch_size=batch_size,sampler=sampler,
        num_workers=workers,pin_memory=True,persistent_workers=False,
        **({"prefetch_factor":2} if workers else {}))
    valid_loader=DataLoader(validation,batch_size=batch_size,num_workers=0)
    steps=min(cfg["steps"],len(loader))
    wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False) if world>1 else model
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    encoder_ids={id(p) for p in model.core.encoder.parameters()}
    optimizer=torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.requires_grad and id(p) in encoder_ids],lr=cfg["encoder_lr"],ratio=cfg["encoder_lr"]/cfg["lr"]),
        dict(params=[p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids],lr=cfg["lr"],ratio=1.)],weight_decay=cfg["weight_decay"])
    best=float("inf");updates=0;begin=0
    if (out/"training.pt").is_file():
        state=torch.load(out/"training.pt",map_location="cpu",weights_only=False)
        b.require_finite_state(state)
        if state["fingerprint"]!=cfg["fingerprint"]:raise ValueError("Resume source/config changed")
        model.load_state_dict(state["model"]);ema.load_state_dict(state["ema"]);optimizer.load_state_dict(state["optimizer"])
        best,updates,begin=state["best"],state["updates"],state["epoch"]
    start=time.time();last_epoch=60.
    stop_at=min(start+cfg["train_hours"]*3600/len(cfg["models"]),
                cfg["session_started"]+(cfg["total_hours"]-1.5)*3600)
    resource=dict(rank=rank,world_size=world,gpu=torch.cuda.get_device_name(device),
        model=spec,batch_per_gpu=batch_size,measured_forward_backward_seconds=measured,
        steps=steps,loader_workers=workers,
        train_records=len(training),validation_records=len(validation),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),precision="float32")
    b.save_json(out/f"resources_{rank}.json",resource)
    print("Crop training resources:",json.dumps(resource),flush=True)
    for epoch in range(begin,cfg["epochs"]):
        s.sam_remaining(cfg,120)
        before=time.time();training.epoch=epoch;sampler.set_epoch(epoch);wrapped.train()
        final_epoch=epoch+1==cfg["epochs"] or before+last_epoch*2>stop_at
        total=0.;torch.cuda.reset_peak_memory_stats(device)
        for step,batch in enumerate(loader):
            if step>=steps:break
            progress=max((epoch+step/max(steps,1))/cfg["epochs"],min(1.,(time.time()-start)/max(stop_at-start,1)))
            lr=cfg["lr"]*min(1.,(updates+1)/max(steps*2,1))*(.03+.97*(1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups:group["lr"]=lr*group["ratio"]
            optimizer.zero_grad(set_to_none=True)
            logits,quality=wrapped(batch["image"].to(device).float())
            loss=crop_loss(logits,quality,batch,device)
            b.require_finite(loss,"crop loss",world>1);loss.backward()
            norm=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=False)
            b.require_finite(norm,"crop gradient norm",world>1)
            optimizer.step();updates+=1
            b.require_finite(torch.stack([p.detach().norm() for p in model.parameters()]),"crop parameters",world>1)
            b.update_ema(ema,model,min(cfg["ema_decay"],(1+updates)/(10+updates)))
            total+=float(loss.detach())
        evaluate=epoch==0 or (epoch+1)%cfg["validation_every"]==0 or final_epoch
        val_sum=0.;val_count=0
        if evaluate:
            with torch.inference_mode():
                for batch in valid_loader:
                    loss=crop_loss(*ema(batch["image"].to(device).float()),batch,device)
                    b.require_finite(loss,"crop validation loss",world>1)
                    n=len(batch["nviews"]);val_sum+=float(loss)*n;val_count+=n
        values=torch.tensor([total/max(steps,1),val_sum,val_count],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(values)
        values=values.cpu().numpy();val=float(values[1]/max(values[2],1)) if evaluate else None
        improved=evaluate and val<best
        if improved:best=val
        if rank==0:
            checkpoint=dict(state=ema.state_dict(),epoch=epoch+1,spec=spec,validation_loss=val,
                fingerprint=cfg["fingerprint"],precision="float32",prompt="noisy filled box; no GT mask boundary input")
            if improved:b.checkpoint_save(out/"best.pt",checkpoint)
            b.checkpoint_save(out/"last.pt",checkpoint)
            b.checkpoint_save(out/"training.pt",dict(model=model.state_dict(),ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),best=best,updates=updates,epoch=epoch+1,fingerprint=cfg["fingerprint"]))
            row=dict(epoch=epoch+1,training_loss=float(values[0]/world),validation_loss=val,
                seconds=time.time()-before,updates=updates,lr=lr,peak_vram_gib=torch.cuda.max_memory_allocated(device)/2**30)
            with (out/"history.jsonl").open("a",encoding="utf8") as stream:stream.write(json.dumps(row)+"\n")
            print(f"Crop member {member}:",json.dumps(row),flush=True)
        last_epoch=time.time()-before
        stop=torch.tensor(int(final_epoch or time.time()+last_epoch*1.5>stop_at),device=device)
        if world>1:dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if stop.item():break
    if rank==0:
        b.save_json(out/"complete.json",dict(epochs=epoch+1,updates=updates,best_crop_validation_loss=best,
            seconds=time.time()-start,best_sha256=b.file_sha(out/"best.pt")))
        (out/"training.pt").unlink(missing_ok=True)


def launch_crop(cfg,task,extra=None,distributed=False):
    command=[sys.executable]
    if distributed and len(cfg["gpu_ids"])>1:
        command += ["-m","torch.distributed.run","--standalone","--nnodes=1",f"--nproc_per_node={len(cfg['gpu_ids'])}"]
    command += [str(CROP_WORKER),"--config",str(Path(cfg["run"])/"config.json"),"--task",task]
    command += list(map(str,extra or []))
    print("Launching:"," ".join(command),flush=True)
    return subprocess.Popen(command,start_new_session=os.name!="nt")


def train_crops(cfg):
    run=Path(cfg["run"])
    for member,spec in enumerate(cfg["models"]):
        out=run/f"model_{member}";out.mkdir(exist_ok=True)
        if (out/"complete.json").is_file():continue
        if not (out/"initial.pt").is_file():
            b.seed_all(cfg["seed"]+member*100_003)
            model=BoxCropUNet(spec,True)
            b.require_finite_state(model.state_dict());b.checkpoint_save(out/"initial.pt",model.state_dict())
            del model
        s.await_sam(cfg,[launch_crop(cfg,"train",["--member",member],True)])
        (out/"initial.pt").unlink(missing_ok=True)
    return cfg


def predict_crop_batch(models,image,device,tta):
    predictions=[];qualities=[]
    with torch.inference_mode():
        for model in models:
            total=torch.zeros((len(image),1,*image.shape[-2:]),device=device)
            quality_sum=torch.zeros(len(image),device=device)
            for dims in [(),(-1,),(-2,),(-2,-1)][:tta]:
                logits,quality=model(image.flip(dims) if dims else image)
                probability=logits.float().sigmoid()
                if dims:probability=probability.flip(dims)
                total+=probability/tta;quality_sum+=quality.float()/tta
            b.require_finite(total,"crop inference mask");b.require_finite(quality_sum,"crop inference quality")
            predictions.append(total[:,0].cpu().numpy());qualities.append(quality_sum.cpu().numpy())
    values={f"m{i}":(p,q) for i,(p,q) in enumerate(zip(predictions,qualities))}
    values["avg"]=(np.mean(predictions,axis=0),np.mean(qualities,axis=0))
    return values


def supported_pieces(probability,seed,threshold):
    pieces=[]
    for box,mask in b.components(probability,threshold,20,96):
        a,c,d,e=box;support=float(seed[c:e,a:d][mask].sum())
        if support>max(1.,float(seed.sum())*.02):pieces.append((support,box,mask))
    pieces=sorted(pieces,key=lambda item:-item[0])[:3]
    result=[(box,mask) for _,box,mask in pieces]
    if len(pieces)>1:
        x0=min(box[0] for _,box,_ in pieces);y0=min(box[1] for _,box,_ in pieces)
        x1=max(box[2] for _,box,_ in pieces);y1=max(box[3] for _,box,_ in pieces)
        union=np.zeros((y1-y0,x1-x0),bool)
        for _,(a,c,d,e),mask in pieces:union[c-y0:e-y0,a-x0:d-x0]|=mask
        result.append(([x0,y0,x1,y1],union))
    return result


def infer_crop_worker(config_path,part,slot):
    cfg=b.load_json(config_path);run=Path(cfg["run"]);manifest=b.load_json(run/"manifest.json")
    device=torch.device(f"cuda:{cfg['gpu_ids'][slot]}");torch.cuda.set_device(device)
    models=[]
    for member,spec in enumerate(cfg["models"]):
        checkpoint=torch.load(run/f"model_{member}"/"best.pt",map_location="cpu",weights_only=False)
        if checkpoint["fingerprint"]!=cfg["fingerprint"]:raise ValueError("Wrong crop checkpoint")
        b.require_finite_state(checkpoint)
        model=BoxCropUNet(spec,False);model.load_state_dict(checkpoint["state"])
        models.append(model.to(device).eval().requires_grad_(False))
    quality=s.prior_quality(cfg)
    output=run/"candidates"/part;output.mkdir(parents=True,exist_ok=True)
    variants=[f"m{i}" for i in range(len(models))]+["avg"]
    sources=manifest["splits"][part][slot::len(cfg["gpu_ids"])]
    for index,source in enumerate(sources):
        s.sam_remaining(cfg,60)
        target=output/(source["stem"]+".json")
        if target.is_file():continue
        image=cv2.imread(source["cache"])
        with np.load(source["prior_map"]) as data:probability=data["prob"].astype(np.float32)/255
        base,seeds=s.propose_seeds(source,cfg,manifest,quality,probability,image)
        for c in base:c["features"] += [0.,0.,1.,0.]
        sets={"baseline":base,**{f"{variant}_{threshold}":[] for variant in variants for threshold in cfg["mask_thresholds"]}}
        for start in range(0,len(seeds),cfg["inference_batch"]):
            current=seeds[start:start+cfg["inference_batch"]]
            rois=[s.square_roi(c["box"],cfg) for c in current]
            pixels=np.stack([box_input(image[:,:,0],c["box"],roi,cfg["crop_size"]) for c,roi in zip(current,rois)])
            predicted=predict_crop_batch(models,torch.from_numpy(pixels).to(device),device,cfg["tta"])
            for variant,(maps,quality_scores) in predicted.items():
                for candidate,roi,mask,q in zip(current,rois,maps,quality_scores):
                    x0,y0,x1,y1=roi
                    local=cv2.resize(mask,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
                    seed=s.put_seed_in_roi(candidate,roi,probability[0])
                    for threshold in cfg["mask_thresholds"]:
                        for box,piece in supported_pieces(local,seed,threshold):
                            a,c,d,e=box;value=b.make_candidate([x0+a,y0+c,x0+d,y0+e],piece,probability[0],probability[1],image)
                            value["features"] += [float(np.clip(q,0,1)),float(local[c:e,a:d][piece].mean()),
                                float(seed[c:e,a:d][piece].sum()/max(seed.sum(),1e-6)),1.]
                            sets[f"{variant}_{threshold}"].append(value)
        if part=="calibration":
            for candidates in sets.values():
                for candidate,value in zip(candidates,b.candidate_targets(source,candidates)):candidate["target"]=float(value)
        b.save_json(target,sets)
        print(f"Crop GPU {device.index}: {part} {index+1}/{len(sources)}, {len(seeds)} boxes",flush=True)


def predict_crops(cfg,part):
    s.await_sam(cfg,[launch_crop(cfg,"infer",["--part",part,"--slot",slot]) for slot in range(len(cfg["gpu_ids"]))])
    return Path(cfg["run"])/"candidates"/part


def select_crop_instances(sets,recipe,quality=None,oof=False):
    if recipe["family"]=="prior":return [c["rle"] for c in sets["baseline"]]
    translated={"baseline":sets["baseline"],
        f"refined_{recipe['threshold']}":sets[f"{recipe['variant']}_{recipe['threshold']}"]}
    return s.select_refiner_instances(translated,recipe,quality,oof)


def calibrate_crops(cfg,folder):
    model,data=s.fit_refiner_quality(cfg,folder)
    run=Path(cfg["run"]);manifest=b.load_json(run/"manifest.json")
    variants=[f"m{i}" for i in range(len(cfg["models"]))]+["avg"]
    recipes=[dict(family=family,variant=variant,threshold=t,cutoff=q,min_area=area)
        for family in ("refined","fusion") for variant in variants for t in cfg["mask_thresholds"]
        for q in (.15,.20,.25,.30,.35) for area in (40,100,200)]
    def score(source):
        s.sam_remaining(cfg,600);sets=data[source["stem"]]
        for values in sets.values():
            for candidate in values:candidate["_decoded"]=b.decode(candidate["local_rle"])
        counts=np.zeros((len(recipes),4),np.float64)
        for i,recipe in enumerate(recipes):
            pred=select_crop_instances(sets,recipe,model,True)
            for view in source["views"]:counts[i]+=b.metric_counts(view["rles"],pred)
        for values in sets.values():
            for candidate in values:candidate.pop("_decoded",None)
        return counts
    counts=sum(b.bounded_map(score,manifest["splits"]["calibration"],4))
    grid=sorted([dict(**recipe,**b.summarize(count)) for recipe,count in zip(recipes,counts)],key=lambda row:-row["PQ"])
    baseline=np.zeros(4,np.float64)
    for source in manifest["splits"]["calibration"]:
        pred=select_crop_instances(data[source["stem"]],dict(family="prior"))
        for view in source["views"]:baseline+=b.metric_counts(view["rles"],pred)
    baseline_pq=b.summarize(baseline)["PQ"]
    chosen={key:grid[0][key] for key in ("family","variant","threshold","cutoff","min_area")}
    if cfg["mode"]=="full" and grid[0]["PQ"]<baseline_pq+cfg["minimum_calibration_gain"]:
        chosen=dict(family="prior")
    b.save_json(run/"calibration_grid.json",grid)
    b.save_json(run/"selected_refinement.json",dict(recipe=chosen,refiner_best=grid[0],
        prior_calibration_PQ=baseline_pq,minimum_gain=cfg["minimum_calibration_gain"],report_used=False))
    audit=b.load_json(run/"quality_audit.json")
    audit.update(features=list(b.FEATURES)+["crop_quality","crop_probability","seed_coverage","is_refined"],model_family="box-crop-cnn")
    b.save_json(run/"quality_audit.json",audit)
    print("Selected crop recipe:",json.dumps(chosen),"| crop PQ",grid[0]["PQ"],"| prior",baseline_pq,flush=True)
    return chosen


def baseline_candidates(cfg,part):
    run=Path(cfg["run"]);manifest=b.load_json(run/"manifest.json");quality=s.prior_quality(cfg)
    output=run/"candidates"/part;output.mkdir(parents=True,exist_ok=True)
    for source in manifest["splits"][part]:
        b.save_json(output/(source["stem"]+".json"),dict(baseline=s.prior_instances(source,manifest,quality)))
    return output


def evaluate_crops(cfg,folder,recipe):
    from collections import defaultdict
    run=Path(cfg["run"]);manifest=b.load_json(run/"manifest.json")
    model=s.load_quality(run/"refiner_quality.joblib")
    dates=defaultdict(lambda:np.zeros((2,4),np.float64));rows=[]
    for source in manifest["splits"]["report"]:
        sets=b.load_json(folder/(source["stem"]+".json"))
        predictions=select_crop_instances(sets,recipe,model)
        prior=[c["rle"] for c in sets["baseline"]]
        counts=np.zeros((2,4),np.float64)
        for view in source["views"]:
            counts[0]+=b.metric_counts(view["rles"],predictions);counts[1]+=b.metric_counts(view["rles"],prior)
        dates[source["date"]]+=counts
        rows.append(dict(stem=source["stem"],date=source["date"],**b.summarize(counts[0])))
    samples=np.asarray(list(dates.values()));totals=samples.sum(0);rng=np.random.default_rng(cfg["seed"])
    differences=[]
    for _ in range(1000):
        sampled=samples[rng.integers(len(samples),size=len(samples))].sum(0)
        differences.append(b.summarize(sampled[0])["PQ"]-b.summarize(sampled[1])["PQ"])
    result=dict(selected=b.summarize(totals[0]),baseline=b.summarize(totals[1]),recipe=recipe,
        difference_PQ=b.summarize(totals[0])["PQ"]-b.summarize(totals[1])["PQ"],
        paired_date_bootstrap_delta_95ci=list(map(float,np.quantile(differences,[.025,.975]))),
        report_used_for_fitting=False,report_dates_reused_across_experiments=True,
        neural_report_inference_skipped=recipe["family"]=="prior",public_PQ="not measured")
    b.save_json(run/"refiner_report.json",result);b.save_json(run/"report_by_image.json",rows)
    print("Crop held-out report:",json.dumps(result),flush=True)
    return result


def export_crops(cfg,folder,recipe):
    # The shared export owns native-grid, disjointness, ID and RLE audits.
    run=Path(cfg["run"]);translated=run/"export_candidates";translated.mkdir(exist_ok=True)
    manifest=b.load_json(run/"manifest.json")
    for source in manifest["splits"]["test"]:
        sets=b.load_json(folder/(source["stem"]+".json"));value=dict(baseline=sets["baseline"])
        if recipe["family"]!="prior":
            value[f"refined_{recipe['threshold']}"]=sets[f"{recipe['variant']}_{recipe['threshold']}"]
        b.save_json(translated/(source["stem"]+".json"),value)
    result=s.export_refiner(cfg,translated,recipe)
    timing=b.load_json(run/"session_timing.json");timing.pop("decoder_only_training",None)
    timing.update(image_trained_box_crops=True,model_members=len(cfg["models"]))
    b.save_json(run/"session_timing.json",timing)
    return result


def crop_main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    parser.add_argument("--task",choices=["train","infer"],required=True)
    parser.add_argument("--member",type=int,default=0);parser.add_argument("--slot",type=int,default=0)
    parser.add_argument("--part",default="test");args=parser.parse_args()
    try:
        if args.task=="train":train_crop_worker(args.config,args.member)
        else:infer_crop_worker(args.config,args.part,args.slot)
    finally:
        if dist.is_initialized():dist.destroy_process_group()



if __name__=='__main__':
    crop_main()
