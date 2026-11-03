"""Verify scorer equivalence, raster alignment, and completeness of the audit."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from audit_training_labels import counts, encode, annotation_diagnostics, mu

ROOT = Path(__file__).parent
official = json.loads(Path("self-evaluation-notebook.ipynb").read_text(encoding="utf-8"))
namespace = dict(torch=torch, np=np, pd=pd, mask_util=mu)
exec("".join(official["cells"][3]["source"]), namespace)
exec("".join(official["cells"][4]["source"]), namespace)

a = np.zeros((64,64), np.uint8)
a[4:25,5:30] = 1
b = np.zeros_like(a)
b[6:27,7:32] = 1
c = np.zeros_like(a)
c[35:50,35:50] = 1
ra, rb, rc = map(encode, [a,b,c])
cases = [([],[]), ([ra],[]), ([],[ra]), ([ra],[ra]), ([ra],[rb]),
         ([ra,rc],[ra,rb]), ([ra],[ra,rb])]
for pred, gt in cases:
    matrix = np.asarray(mu.iou(pred,gt,[0]*len(gt))).T if pred and gt else np.zeros((len(gt),len(pred)))
    expected = namespace["get_pq_score"](pd.DataFrame([
        dict(iou_matrix=torch.from_numpy(matrix), n_gt=len(gt), n_pred=len(pred))]))
    assert abs(counts(pred,gt)["pq"] - expected) < 1e-12

mask = np.zeros((2048,2048), np.uint8)
mask[7:18,11:39] = 1
row, _ = annotation_diagnostics(None, dict(id="synthetic", spine=[]), encode(mask))
for factor in [2,4,8]:
    small = cv2.resize(mask,(2048//factor,2048//factor),interpolation=cv2.INTER_LINEAR)
    reconstructed = cv2.resize(small,(2048,2048),interpolation=cv2.INTER_NEAREST)
    expected = (mask & reconstructed).sum() / (mask | reconstructed).sum()
    assert abs(row[f"grid{factor}_iou"] - expected) < 1e-12

summary = json.loads((ROOT / "audit/summary.json").read_text())
sources = pd.read_csv(ROOT / "audit/physical_image_audit.csv")
annotations = pd.read_csv(ROOT / "audit/annotation_audit.csv")
assert summary["complete_training_audit"]
assert len(sources) == summary["train_physical_images"] == 707
assert len(annotations) == summary["annotations"] == 8199
assert sources.views.sum() == summary["annotator_records"] == 1154
assert annotations.annotation_id.is_unique
assert not annotations.isna()[["area", "grid2_iou", "grid4_iou", "grid8_iou"]].any().any()
for method, values in summary["oracle_diagnostics"].items():
    assert len(pd.read_csv(ROOT / "audit" / (method + ".csv"))) == 1154
print("PASS: 7 official-scorer cases, 3 raster-alignment cases, complete 707-image/8199-annotation audit.")
