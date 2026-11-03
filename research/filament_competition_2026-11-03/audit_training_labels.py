"""Training-label diagnostics for the Solar Filament Segmentation Challenge.

This reads only the supplied competition TRAIN annotations. Test files are
counted by filename; no external/test annotations are used. Oracle masks use
training ground truth to isolate label-conversion effects: their scores are
diagnostics, never validation scores, leaderboard predictions or model ceilings.
Run from the repository root with its .venv Python.
"""
import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from pycocotools.coco import COCO

cv2.setNumThreads(1)
SIZE = 2048


def encode(mask):
    return mu.encode(np.asfortranarray(mask, dtype=np.uint8))


def counts(predicted, targets, one_to_one=False):
    n, m = len(predicted), len(targets)
    iou = np.asarray(mu.iou(predicted, targets, [0] * m)).T if n and m else np.zeros((m, n))
    hit = iou > 0.5
    if one_to_one:
        candidates = sorted(((float(iou[g, p]), int(g), int(p)) for g, p in np.argwhere(hit)), reverse=True)
        used_g, used_p, total = set(), set(), 0.0
        for value, g, p in candidates:
            if g not in used_g and p not in used_p:
                used_g.add(g)
                used_p.add(p)
                total += value
        tp, fp, fn = len(used_g), n - len(used_p), m - len(used_g)
    else:
        tp = int(hit.sum())
        fp, fn = int((hit.sum(0) == 0).sum()), int((hit.sum(1) == 0).sum())
        total = float(iou[hit].sum())
    denominator = tp + .5 * (fp + fn)
    return dict(tp=tp, fp=fp, fn=fn, sum_iou=total, pq=total / denominator if denominator else 0.0,
                ambiguous_predictions=int((hit.sum(0) > 1).sum()),
                ambiguous_targets=int((hit.sum(1) > 1).sum()), n_pred=n, n_gt=m)


def pooled(rows):
    if not rows:
        return dict(PQ=0.0, SQ=0.0, RQ=0.0, TP=0, FP=0, FN=0, records=0)
    tp, fp, fn = (sum(r[k] for r in rows) for k in ["tp", "fp", "fn"])
    total = sum(r["sum_iou"] for r in rows)
    denominator = tp + .5 * (fp + fn)
    return dict(PQ=total / denominator if denominator else 0.0, SQ=total / tp if tp else 0.0,
                RQ=tp / denominator if denominator else 0.0, TP=tp, FP=fp, FN=fn,
                records=len(rows), mean_record_PQ=float(np.mean([r["pq"] for r in rows])))


def component_rles(binary):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    rles, areas = [], []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        full = np.zeros(binary.shape, np.uint8)
        full[y:y+h, x:x+w] = labels[y:y+h, x:x+w] == label
        rles.append(encode(full))
        areas.append(area)
    return rles, np.asarray(areas)


def annotation_diagnostics(coco, annotation, rle):
    mask = mu.decode(rle)
    x, y, w, h = (int(v) for v in mu.toBbox(rle))
    area = int(mask.sum())
    if not area:
        return dict(annotation_id=annotation["id"], area=0), mask
    crop = mask[y:y+h, x:x+w]
    _, _, stats, _ = cv2.connectedComponentsWithStats(crop, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    row = dict(annotation_id=annotation["id"], area=area,
               components=int(len(sizes)), components_ge20=int((sizes >= 20).sum()),
               largest_component_fraction=float(sizes.max() / area), bbox_width=w, bbox_height=h)
    spine = np.asarray(annotation.get("spine", []), dtype=float).reshape(-1, 2)
    length = float(np.linalg.norm(np.diff(spine, axis=0), axis=1).sum()) if len(spine) > 1 else 0.0
    row.update(spine_length=length, area_per_spine_length=area / length if length > 0 else None)
    # Raster-grid diagnostic: binary target downsampling followed by nearest
    # reconstruction. This is not a bound on a learned decoder's performance.
    for factor in [2, 4, 8]:
        x0, y0 = x // factor * factor, y // factor * factor
        x1, y1 = min(SIZE, ((x+w+factor-1)//factor)*factor), min(SIZE, ((y+h+factor-1)//factor)*factor)
        aligned = mask[y0:y1, x0:x1]
        small = cv2.resize(aligned, ((x1-x0)//factor, (y1-y0)//factor), interpolation=cv2.INTER_LINEAR)
        restored = cv2.resize(small, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST)
        inter = int((aligned & restored).sum())
        union = int((aligned | restored).sum())
        row[f"grid{factor}_iou"] = inter / union if union else 0.0
        row[f"grid{factor}_area_retained"] = int(restored.sum()) / area
    return row, mask


def audit_source(coco, item):
    name, ids = item
    targets, annotations, unions = {}, [], []
    overlapping_pairs = confusing_pairs = 0
    full_union = np.zeros((SIZE, SIZE), np.uint8)
    metrics = defaultdict(list)
    for image_id in ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        targets[image_id] = []
        union = np.zeros_like(full_union)
        for ann in anns:
            rle = coco.annToRLE(ann)
            if int(mu.area(rle)) == 0:
                continue
            targets[image_id].append(rle)
            row, mask = annotation_diagnostics(coco, ann, rle)
            row.update(file_name=name, image_id=image_id)
            annotations.append(row)
            union |= mask
        unions.append(union)
        full_union |= union
        self_iou = np.asarray(mu.iou(targets[image_id], targets[image_id], [0] * len(targets[image_id])))
        if self_iou.size:
            upper = np.triu(self_iou, 1)
            overlapping_pairs += int((upper > 0).sum())
            confusing_pairs += int((upper > .5).sum())
        predictions, areas = component_rles(union)
        filtered = [p for p, a in zip(predictions, areas) if a >= 20]
        metrics["single_annotator_union_cc20"].append(dict(file_name=name, image_id=image_id,
                                                           **counts(filtered, targets[image_id])))
    predictions, areas = component_rles(full_union)
    for min_area in [1, 20, 80, 200]:
        filtered = [p for p, a in zip(predictions, areas) if a >= min_area]
        for image_id in ids:
            metrics[f"all_annotators_union_cc{min_area}"].append(
                dict(file_name=name, image_id=image_id, **counts(filtered, targets[image_id])))
            if min_area == 20:
                metrics["all_annotators_union_cc20_one_to_one"].append(
                    dict(file_name=name, image_id=image_id, **counts(filtered, targets[image_id], one_to_one=True)))
    human_rows = []
    for i, j in itertools.permutations(ids, 2):
        human_rows.append(dict(file_name=name, predicted_view=i, target_view=j,
                               **counts(targets[i], targets[j])))
    # One whole-image union is also scored at the pixel level per annotator.
    dice = [2 * int((union & full_union).sum()) / max(1, int(union.sum()) + int(full_union.sum())) for union in unions]
    source = dict(file_name=name, views=len(ids), instances=sum(len(v) for v in targets.values()),
                  union_pixels=int(full_union.sum()), union_components=len(predictions),
                  union_components_ge20=int((areas >= 20).sum()),
                  union_components_ge200=int((areas >= 200).sum()),
                  union_vs_annotator_mean_dice=float(np.mean(dice)),
                  overlapping_gt_pairs=overlapping_pairs, gt_pairs_iou_gt_half=confusing_pairs,
                  human_mean_pair_pq=float(np.mean([r["pq"] for r in human_rows])) if human_rows else None)
    return dict(source=source, annotations=annotations, human_rows=human_rows, metrics=dict(metrics))


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if len(a) == 0:
        return {}
    return dict(count=len(a), mean=float(a.mean()), **{f"p{p:02d}": float(np.percentile(a, p)) for p in [5,25,50,75,95]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="MAGFilo_1.0_Kaggle_2026")
    parser.add_argument("--output", default=str(Path(__file__).parent / "audit"))
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    data, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    annotation_path = data / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = COCO(str(annotation_path))
    groups = defaultdict(list)
    for image_id, metadata in coco.imgs.items():
        assert (metadata["height"], metadata["width"]) == (SIZE, SIZE)
        groups[metadata["file_name"]].append(image_id)
    tasks = sorted(groups.items())
    if args.max_images:
        tasks = tasks[:args.max_images]
    results = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        # executor.map queues small filename/ID tasks only; worker results contain
        # scalar tables and compressed RLEs are released before returning.
        for i, result in enumerate(pool.map(lambda item: audit_source(coco, item), tasks), 1):
            results.append(result)
            if i % 25 == 0 or i == len(tasks):
                print(f"Audited {i}/{len(tasks)} images in {time.perf_counter()-start:.1f}s", flush=True)
    source_rows = [r["source"] for r in results]
    annotation_rows = [a for r in results for a in r["annotations"]]
    human_rows = [a for r in results for a in r["human_rows"]]
    methods = defaultdict(list)
    for result in results:
        for key, rows in result["metrics"].items():
            methods[key].extend(rows)
    pd.DataFrame(source_rows).to_csv(output / "physical_image_audit.csv", index=False)
    pd.DataFrame(annotation_rows).to_csv(output / "annotation_audit.csv", index=False)
    pd.DataFrame(human_rows).to_csv(output / "human_agreement.csv", index=False)
    for name, rows in methods.items():
        pd.DataFrame(rows).to_csv(output / (name + ".csv"), index=False)
    ann = pd.DataFrame(annotation_rows)
    sources = pd.DataFrame(source_rows)
    train_names = set(groups)
    test_names = {p.name for p in (data / "test/test_images").iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}}
    shared_dates = {n[:8] for n in train_names} & {n[:8] for n in test_names}
    summary = dict(annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                   seconds=time.perf_counter()-start, audited_images=len(tasks),
                   complete_training_audit=len(tasks)==len(groups),
                   train_physical_images=len(groups), annotator_records=len(coco.imgs),
                   annotations=len(coco.anns), test_images=len(test_names),
                   filename_overlap=len(train_names & test_names),
                   train_unique_dates=len({n[:8] for n in train_names}),
                   test_unique_dates=len({n[:8] for n in test_names}),
                   shared_train_test_dates=len(shared_dates),
                   test_images_sharing_training_date=sum(n[:8] in shared_dates for n in test_names),
                   views_per_image=dict(Counter(len(v) for v in groups.values())),
                   train_sites=dict(Counter(Path(n).stem[-2:] for n in train_names)),
                   test_sites=dict(Counter(Path(n).stem[-2:] for n in test_names)),
                   train_years=dict(sorted(Counter(n[:4] for n in train_names).items())),
                   test_years=dict(sorted(Counter(n[:4] for n in test_names).items())),
                   instance_area=distribution(ann.area),
                   fraction_area_lt1000=float((ann.area < 1000).mean()),
                   fraction_area_lt200=float((ann.area < 200).mean()),
                   area_per_spine_length=distribution(ann.area_per_spine_length),
                   fraction_instances_multiple_components=float((ann.components > 1).mean()),
                   fraction_instances_multiple_components_ge20=float((ann.components_ge20 > 1).mean()),
                   single_annotation_largest_component_fraction=distribution(ann.largest_component_fraction),
                   union_foreground_percent=distribution(sources.union_pixels / SIZE**2 * 100),
                   source_union_vs_annotator_dice=distribution(sources.union_vs_annotator_mean_dice),
                   sources_with_gt_iou_gt_half=int((sources.gt_pairs_iou_gt_half > 0).sum()),
                   gt_pairs_iou_gt_half=int(sources.gt_pairs_iou_gt_half.sum()),
                   human_agreement=pooled(human_rows),
                   human_mean_physical_image_PQ=float(sources.human_mean_pair_pq.dropna().mean()),
                   oracle_diagnostics={key: pooled(rows) for key, rows in methods.items()},
                   oracle_by_annotation_count={key: {
                       "single_annotator_sources": pooled([r for r in rows if len(groups[r["file_name"]]) == 1]),
                       "multiple_annotator_sources": pooled([r for r in rows if len(groups[r["file_name"]]) > 1])}
                       for key, rows in methods.items()},
                   raster_grid_diagnostics={str(f): dict(
                       iou=distribution(ann[f"grid{f}_iou"]),
                       fraction_iou_gt_half=float((ann[f"grid{f}_iou"] > .5).mean()),
                       area_retained=distribution(ann[f"grid{f}_area_retained"])) for f in [2,4,8]})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
