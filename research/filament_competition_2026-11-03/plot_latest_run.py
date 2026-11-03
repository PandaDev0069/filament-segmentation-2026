"""Plot measured training, calibration and error diagnostics from results.zip."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).parent / "latest_run"
summary = json.loads((root / "analysis_summary.json").read_text())
history = pd.read_csv(root / "training_history.csv")
sizes = pd.read_csv(root / "calibration_recall_by_area.csv")
fits = pd.read_csv(root / "report_by_disk_fit.csv")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
axes[0,0].plot(history.epoch, history["metrics/mAP50-95(M)"], color="#39769B")
axes[0,0].axvline(36, ls="--", color="#BC6D52", label="Best logged mask mAP: epoch 36")
axes[0,0].set(xlabel="Epoch", ylabel="Mask mAP50-95", title="Training improved beyond epoch 5")
axes[0,0].legend(fontsize=8)
modes = summary["calibration_mode_best"]
axes[0,1].bar(["D4 hysteresis", "D4 + morphology", "Single-view union"],
              [r["PQ"] for r in modes], color=["#39769B", "#BC6D52", "#B8C0C7"])
axes[0,1].set(ylabel="Calibration PQ", ylim=(0,.5), title="Best searched rule in each mode (selection scores)")
for i,r in enumerate(modes):
    axes[0,1].text(i,r["PQ"]+.01,f"{r['PQ']:.3f}",ha="center")
axes[1,0].bar(sizes.area_group,sizes.recall,color="#39769B")
axes[1,0].set(xlabel="Ground-truth instance area (native pixels)",ylabel="Matched-instance recall",
              ylim=(0,.85),title="Small filaments are frequently missed (calibration)")
for i,r in sizes.iterrows():
    axes[1,0].text(i,r.recall+.02,f"{r.recall:.1%}\nn={r.instances}",ha="center",fontsize=8)
fits = fits.sort_values("group")
axes[1,1].bar(["Fit failed / raw fallback", "Fit succeeded"],fits.PQ,color=["#BC6D52", "#39769B"])
axes[1,1].set(ylabel="Held-out report PQ",ylim=(0,.5),
              title="Disk-fit status is associated with performance")
for i,r in fits.reset_index(drop=True).iterrows():
    axes[1,1].text(i,r.PQ+.01,f"{r.PQ:.3f}\n{int(r.images)} images",ha="center",fontsize=9)
fig.suptitle("Latest run: measured weaknesses; disk-fit association is not proof of causation",fontsize=13)
fig.savefig(root / "run_diagnostics.png",dpi=150)
plt.close(fig)
print(root / "run_diagnostics.png")
