"""Render research figures from measured training-label diagnostics."""
from pathlib import Path
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools.coco import COCO

ROOT = Path(__file__).parent
summary = json.loads((ROOT / "audit/summary.json").read_text())
sources = pd.read_csv(ROOT / "audit/physical_image_audit.csv")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
views = summary["views_per_image"]
axes[0].bar([1, 2, 3], [views[str(i)] for i in [1, 2, 3]], color=["#547E9D", "#E3A246", "#B86655"])
axes[0].set(xticks=[1,2,3], xlabel="Independent annotators per physical image", ylabel="Images",
            title="707 unique training images")
for i in [1, 2, 3]:
    axes[0].text(i, views[str(i)] + 6, str(views[str(i)]), ha="center")
methods = summary["oracle_diagnostics"]
values = [methods["single_annotator_union_cc20"]["PQ"], methods["all_annotators_union_cc20"]["PQ"]]
axes[1].bar([0, 1], values, color=["#547E9D", "#B86655"])
axes[1].set(xticks=[0,1], xticklabels=["Own annotator's\nunion + components", "All annotators'\nunion + components"],
            ylim=(0, 1.08), ylabel="Pooled PQ against original annotator instances",
            title="Perfect-label conversion diagnostic")
for i, v in enumerate(values):
    axes[1].text(i, v + .02, f"{v:.4f}", ha="center")
grid = summary["raster_grid_diagnostics"]
factors = [2,4,8]
median = np.array([grid[str(i)]["iou"]["p50"] for i in factors])
low = np.array([grid[str(i)]["iou"]["p25"] for i in factors])
high = np.array([grid[str(i)]["iou"]["p75"] for i in factors])
axes[2].errorbar(factors, median, yerr=[median-low,high-median], fmt="o-", capsize=5, color="#547E9D")
axes[2].axhline(.5, linestyle="--", color="#B86655", label="IoU hit threshold")
axes[2].set(xticks=factors, ylim=(.4,1), xlabel="Binary target grid downsampling factor",
            ylabel="IoU after nearest-neighbor reconstruction",
            title="Raster-grid diagnostic (median and IQR)")
axes[2].legend(fontsize=8)
fig.suptitle("Training-label measurements: these are not trained-model or leaderboard scores", fontsize=13)
fig.savefig(ROOT / "training_label_diagnostics.png", dpi=160)
plt.close(fig)

# Illustrative three-annotator case near the 25th percentile of union agreement,
# with enough annotated objects to show differences in both extent and presence.
multi = sources[(sources.views == 3) & (sources.instances >= 10)].copy()
target = float(multi.union_vs_annotator_mean_dice.quantile(.25))
row = multi.iloc[np.argmin(np.abs(multi.union_vs_annotator_mean_dice - target))]
coco = COCO("MAGFilo_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
ids = [i for i,m in coco.imgs.items() if m["file_name"] == row.file_name]
image = cv2.imread(str(Path("MAGFilo_1.0_Kaggle_2026/train/train_images") / row.file_name), 0)
masks = []
for image_id in ids:
    union = np.zeros(image.shape, np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
        union |= coco.annToMask(ann).astype(np.uint8)
    masks.append(union)
votes = np.stack(masks).sum(0)
disagreement = ((votes > 0) & (votes < len(masks))).astype(np.float32)
density = cv2.GaussianBlur(disagreement, (0,0), sigmaX=60)
y, x = np.unravel_index(int(density.argmax()), density.shape)
x0, y0 = int(np.clip(x-256, 0, image.shape[1]-512)), int(np.clip(y-256, 0, image.shape[0]-512))
window = (slice(y0, y0+512), slice(x0, x0+512))
panels = masks + [(votes > 0).astype(np.uint8)]
fig, axes = plt.subplots(2, len(panels), figsize=(16, 8), constrained_layout=True)
for index, mask in enumerate(panels):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    color = np.array([255, 160, 30] if index == len(masks) else [40, 220, 210])
    rgb[mask > 0] = (rgb[mask > 0] * .25 + color * .75).astype(np.uint8)
    axes[0,index].imshow(rgb)
    axes[1,index].imshow(rgb[window], interpolation="nearest")
    title = f"Annotator view {index+1}: {len(coco.getAnnIds(imgIds=[ids[index]]))} instances" if index < len(masks) else "All-annotator union"
    axes[0,index].set_title(title)
    axes[1,index].set_title("Same native 512 x 512 region")
    for ax in axes[:,index]:
        ax.set_axis_off()
fig.suptitle(f"{row.file_name}: independent targets differ in presence and boundaries\n"
             "Illustrative lower-agreement case; selected by a stated rule, not a model prediction", fontsize=13)
fig.savefig(ROOT / "annotation_disagreement_example.png", dpi=150)
plt.close(fig)
(ROOT / "figure_metadata.json").write_text(json.dumps(dict(
    example_file=row.file_name, view_ids=ids, union_mean_dice=float(row.union_vs_annotator_mean_dice),
    crop_xywh=[x0,y0,512,512], selection="three views, >=10 annotations, nearest to 25th percentile of union agreement"), indent=2))
print("Saved training_label_diagnostics.png and annotation_disagreement_example.png")
