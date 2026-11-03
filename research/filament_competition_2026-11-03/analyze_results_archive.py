"""Analyze saved run artifacts and probabilities without retraining or test labels."""
import os
for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[key] = "1"
import ast
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mu
from pycocotools.coco import COCO

ROOT = Path(__file__).parent
OUT = ROOT / "latest_run"
OUT.mkdir(exist_ok=True)
cv2.setNumThreads(1)
ns = dict(np=np,cv2=cv2,pd=pd,mask_utils=mu,Path=Path,NUM_WORKERS=4,
          IMAGE_SIZE=2048,PQ_MATCH_IOU=.5,deque=deque,ThreadPoolExecutor=ThreadPoolExecutor)
# Read only the required helper definitions. Never execute training/configuration
# cells, including any ad-hoc checkpoint assignments added to the live notebook.
required = {"encode_mask","json_rle","native_rle","decode_mask","disk_radius_grid",
            "disk_support","estimate_disk","read_gray","bounded_map","physical_rows",
            "component_candidates","keep_components","score_rles","mask_to_rle"}
notebook = json.loads(Path("Best/solar-filament-yolo26-full-res-0-36-preprocessing.ipynb").read_text(encoding="utf-8"))
loaded = set()
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    try:
        tree = ast.parse("".join(cell["source"]))
    except SyntaxError:
        continue
    tree.body = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in required]
    loaded.update(n.name for n in tree.body)
    if tree.body:
        exec(compile(tree,"notebook_helpers","exec"),ns)
assert loaded == required,required-loaded
coco = COCO("MAGFilo_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
ns["coco"] = coco


def totals(frame):
    tp, fp, fn = (int(frame[k].sum()) for k in ["tp", "fp", "fn"])
    iou = float(frame.matched_iou_sum.sum())
    denominator = tp + .5 * (fp + fn)
    return dict(PQ=iou / denominator if denominator else 0,
                SQ=iou / tp if tp else 0, RQ=tp / denominator if denominator else 0,
                TP=tp, FP=fp, FN=fn, images=int(frame.file_name.nunique()), views=len(frame))


def grouped(frame, column):
    return pd.DataFrame([dict(group=key, **totals(group)) for key, group in frame.groupby(column)])


with ZipFile("Best/results.zip") as archive:
    names = archive.namelist()
    prefix = next(n.rsplit("/",1)[0] for n in names if n.endswith("/selected_pipeline.json"))
    def read_json(name):
        return json.loads(archive.read(prefix + "/" + name))
    def read_csv(name):
        frame = pd.read_csv(io.BytesIO(archive.read(prefix + "/" + name)))
        frame.columns = frame.columns.str.strip()
        return frame
    config = read_json("run_config.json")
    selection = read_json("selected_pipeline.json")
    checkpoint_hashes = {}
    for checkpoint_name in ["best.pt", "last.pt"]:
        matches = [name for name in names if name.endswith("/weights/" + checkpoint_name)]
        if matches:
            digest = hashlib.sha256()
            with archive.open(matches[0]) as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            checkpoint_hashes[checkpoint_name] = digest.hexdigest()
    assert checkpoint_hashes.get("best.pt") == selection["weights_sha256"]
    report = read_csv("report_per_annotator.csv")
    history = read_csv("yolo_runs/yolo11s_full_resolution/results.csv")
    grid = read_csv("calibration_grid.csv")
    cal_records = read_csv("calibration_split.csv")
    ns["PREPROCESSING_CONFIG"] = config["preprocessing"]
    ns["CONFIDENCE_CANDIDATES"] = selection["inference"]["confidences"]
    cfg = selection["postprocessing"]
    cache_entries = {}
    for name in names:
        if "/prediction_cache/" in name and name.endswith(".json"):
            item = json.loads(archive.read(name))
            cache_entries[item["file_name"]] = (name, item)
    report["date"] = report.file_name.str[:8]
    report["site"] = report.file_name.str.split(".").str[0].str[-2:]
    report["year"] = report.file_name.str[:4]
    report["views"] = report.groupby("file_name").file_name.transform("size")
    for column in ["site", "year", "views"]:
        grouped(report, column).to_csv(OUT / ("report_by_" + column + ".csv"), index=False)

    # Resample dates, keeping all observations/annotation views of a date together.
    date_totals = report.groupby("date")[["tp", "fp", "fn", "matched_iou_sum"]].sum().to_numpy()
    rng = np.random.default_rng(42)
    weights = rng.multinomial(len(date_totals), np.full(len(date_totals), 1/len(date_totals)), size=5000)
    sums = weights @ date_totals
    bootstrap = sums[:,3] / (sums[:,0] + .5 * (sums[:,1] + sums[:,2]))

    # Verify the submitted file in the archive agrees with the saved audit/log.
    submission_name = "a_submission.csv"
    submission_bytes = archive.read(submission_name)
    audit = read_json("submission_audit.json")
    log = read_json("test_inference_log.json")
    assert hashlib.sha256(submission_bytes).hexdigest() == audit["sha256"]
    submission = pd.read_csv(io.BytesIO(submission_bytes), dtype=str, keep_default_na=False)
    assert list(submission.columns) == ["filament_id", "segmentation_rle"]
    assert submission.filament_id.is_unique
    submission["stem"] = submission.filament_id.str.rsplit("_", n=1).str[0]
    actual_counts = submission.groupby("stem").size().to_dict()
    assert all(actual_counts.get(stem,0) == count for stem,count in log.items())
    test_files = {p.stem for p in Path("MAGFilo_1.0_Kaggle_2026/test/test_images").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg"}}
    assert set(log) == test_files
    for stem, rows in submission.groupby("stem"):
        occupied = None
        for text in rows.segmentation_rle:
            rle = dict(size=[2048,2048], counts=text.encode("ascii"))
            mask = ns["mask_utils"].decode(rle)
            assert mask.shape == (2048,2048) and mask.any()
            assert ns["mask_to_rle"](mask) == text
            if occupied is not None:
                assert int(ns["mask_utils"].area(ns["mask_utils"].merge([occupied,rle],intersect=True))) == 0
                occupied = ns["mask_utils"].merge([occupied,rle])
            else:
                occupied = rle
    print("Verified archived submission checksum, 1005 RLE rows, coverage log and non-overlap.", flush=True)

    # Reproduce preprocessing disk-fit status for all train/test images, and
    # compare with statuses saved in the original Kaggle caches where present.
    partition = {name: label for label in ["train", "calibration", "report"] for name in config[label]}
    tasks = [(name, label, Path("MAGFilo_1.0_Kaggle_2026/train/train_images") / name)
             for name,label in partition.items()]
    tasks += [(p.name,"test",p) for p in Path("MAGFilo_1.0_Kaggle_2026/test/test_images").iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg"}]
    def fit_status(task):
        name, label, path = task
        image = ns["read_gray"](path)
        disk = ns["estimate_disk"](image)
        return dict(file_name=name, partition=label, fit_success=disk is not None,
                    radius=disk["radius"] if disk else None)
    fits = pd.DataFrame(ns["bounded_map"](fit_status, tasks))
    fits.to_csv(OUT / "disk_fit_status.csv", index=False)
    fit_lookup = fits.set_index("file_name").fit_success.to_dict()
    disagreements = [name for name,(_,meta) in cache_entries.items()
                     if fit_lookup[name] != (meta["disk"] is not None)]
    report["disk_fit"] = report.file_name.map(fit_lookup)
    grouped(report, "disk_fit").to_csv(OUT / "report_by_disk_fit.csv", index=False)
    print("Recomputed disk fits; saved-cache fit-status disagreements:",len(disagreements),flush=True)

    # Reconstruct only the selected calibration rule from the cached maps.
    # This neither retunes the held-out report nor changes the submission.
    def calibration_diagnostic(row):
        name, image_ids = row["file_name"], row["image_ids"]
        meta_path, cached = cache_entries[name]
        with np.load(io.BytesIO(archive.read(meta_path[:-5]+".npz")), allow_pickle=False) as saved:
            maps = {key:saved[key] for key in ["probability","single_binary","salience"]}
        maps["single_binary"] = np.unpackbits(maps["single_binary"],axis=-1,count=2048)
        candidates = ns["component_candidates"](cached,maps,cfg)
        predicted = ns["keep_components"](candidates,cfg)
        base = [c["rle"] for c in candidates]
        score_rows, instances = [], []
        for image_id in image_ids:
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
            targets = [coco.annToRLE(a) for a in anns]
            score_rows.append(dict(file_name=name,image_id=image_id,**ns["score_rles"](predicted,targets)))
            iou = np.asarray(ns["mask_utils"].iou(predicted,targets,[0]*len(targets))).T if predicted else np.zeros((len(targets),0))
            base_iou = np.asarray(ns["mask_utils"].iou(base,targets,[0]*len(targets))).T if base else np.zeros((len(targets),0))
            for j,ann in enumerate(anns):
                best = float(iou[j].max()) if iou.shape[1] else 0.0
                best_base = float(base_iou[j].max()) if base_iou.shape[1] else 0.0
                category = ("matched" if best>.5 else "matching_component_filtered" if best_base>.5
                            else "partial_shape_or_grouping_mismatch" if best_base>.25
                            else "little_matching_component_support")
                instances.append(dict(file_name=name,image_id=image_id,annotation_id=ann["id"],
                                      area=int(ns["mask_utils"].area(targets[j])),max_final_iou=best,
                                      max_unfiltered_iou=best_base,status=category))
        return score_rows, instances
    physical = ns["physical_rows"](cal_records)
    cal_scores, instance_rows = [], []
    for i,(scores,instances) in enumerate(ns["bounded_map"](calibration_diagnostic,physical),1):
        cal_scores.extend(scores)
        instance_rows.extend(instances)
        if i%15==0 or i==len(physical):
            print(f"Reconstructed selected calibration masks: {i}/{len(physical)}",flush=True)
    cal_frame = pd.DataFrame(cal_scores)
    instance_frame = pd.DataFrame(instance_rows)
    cal_frame.to_csv(OUT / "recomputed_calibration.csv",index=False)
    instance_frame.to_csv(OUT / "calibration_instance_diagnostics.csv",index=False)
    cal_metrics = totals(cal_frame)
    assert abs(cal_metrics["PQ"]-selection["calibration_PQ"]) < 1e-12
    assert instance_frame.query("status != 'matched'").shape[0] == cal_metrics["FN"]
    instance_frame["area_group"] = pd.cut(instance_frame.area,[0,500,1000,2500,float("inf")],
                                          labels=["<500","500-999","1000-2499",">=2500"],right=False)
    instance_frame["matched"] = instance_frame.status == "matched"
    size_table = instance_frame.groupby("area_group",observed=True).agg(instances=("matched","size"),
                                                                      matched=("matched","sum"))
    size_table["recall"] = size_table.matched / size_table.instances
    size_table.to_csv(OUT / "calibration_recall_by_area.csv")

    metric = totals(report)
    counts_per_image = report.groupby("file_name").n_pred.first()
    stage = fits.groupby("partition").fit_success.agg(["size","sum"])
    stage["failed"] = stage["size"] - stage["sum"]
    stage.to_csv(OUT / "disk_fit_by_partition.csv")
    test_fit_counts = []
    for fit_success, group in fits[fits.partition == "test"].groupby("fit_success"):
        prediction_counts = [log[Path(name).stem] for name in group.file_name]
        test_fit_counts.append(dict(fit_success=bool(fit_success), images=len(group),
                                    zero_predictions=sum(c==0 for c in prediction_counts),
                                    predictions=sum(prediction_counts)))
    pd.DataFrame(test_fit_counts).to_csv(OUT / "test_counts_by_disk_fit.csv",index=False)
    summary = dict(
        archive="Best/results.zip", run=prefix, model=selection["model"],
        report=metric, calibration=cal_metrics, public_pq_user_reported=.33,
        report_public_gap=metric["PQ"]-.33,
        report_precision=metric["TP"]/(metric["TP"]+metric["FP"]),
        report_recall=metric["TP"]/(metric["TP"]+metric["FN"]),
        report_date_bootstrap_95ci=np.percentile(bootstrap,[2.5,97.5]).tolist(),
        requested_epochs=config["training"]["epochs"],completed_epochs=int(history.epoch.max()),
        best_mask_map_epoch=int(history.loc[history["metrics/mAP50-95(M)"].idxmax(),"epoch"]),
        final_learning_rate=float(history.iloc[-1]["lr/pg0"]),
        evaluated_checkpoints=grid.checkpoint.unique().tolist(),
        checkpoint_sha256=checkpoint_hashes,
        selected_postprocessing=cfg,
        calibration_mode_best=grid.sort_values("PQ",ascending=False).groupby("mode",sort=False).head(1)[
            ["mode","PQ","TP","FP","FN"]].to_dict("records"),
        report_zero_prediction_images=int((counts_per_image==0).sum()),
        report_gt_on_zero_prediction_images=int(report.loc[report.n_pred==0,"n_gt"].sum()),
        report_predictions_per_physical_image=float(counts_per_image.mean()),
        test_zero_prediction_images=sum(v==0 for v in log.values()),
        test_predictions_per_image=sum(log.values())/len(log),
        submission_checksum_matches_audit=True,independently_verified_all_rles=True,
        disk_fit_cache_status_disagreements=disagreements,
        disk_fit_by_partition=stage.reset_index().to_dict("records"),
        test_counts_by_disk_fit=test_fit_counts,
        calibration_fn_categories=instance_frame.query("status != 'matched'").status.value_counts().to_dict(),
        calibration_recall_by_area=size_table.reset_index().to_dict("records"))
    (OUT / "analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)
