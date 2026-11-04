# One-file Kaggle notebook

Upload **[solar-filament-kaggle-fixed.ipynb](solar-filament-kaggle-fixed.ipynb)**
to Kaggle. This is the only file you need to upload.

1. Attach the `filament-segmentation-2026` competition dataset using **Add Input**.
2. Select **GPU T4 x2** and turn **Internet On**.
3. Run all cells in order. `MODE='full'` is already the default.
4. Download **`/kaggle/working/submission.csv`** from the Output/Files panel.

All helper code is inside the notebook. It installs its packages, searches only
the attached `/kaggle/input` data, downloads YOLO11m and generic ImageNet encoders,
trains, validates and exports masks. Refitting is optional and disabled by default.
It automatically creates a worker
script under `/kaggle/working` for multi-GPU execution. No local repository,
companion Python upload, previous results or manually supplied model weights are
required. Training uses two GPUs with four total data-loader workers.

The training-history cell displays grouped YOLO artifact images, CSV loss/metric
curves, and U-Net loss, calibration PQ, learning rate, optimizer updates, GPU memory
and epoch duration. YOLO image generation is enabled for new runs. Missing images
are skipped; existing CSV/JSON logs can still be plotted. Figures are saved under
the run's `training_visuals/` directory. Optional refit histories display after refit.

The method is the native hybrid pipeline from the previous notebook. Packaging
checks execute the inline helpers in an isolated IPython namespace, assemble and
load the generated worker, verify GPU launch arguments and restrict discovery to
attached inputs. New numerical regressions cover the NaN-loss failure observed
in the full run. See the [November 4 repair report](repair_2026_11_04/REPAIR_REPORT.md).
Pinned packages were also checked against the public PyPI registry.

Full training on Kaggle T4 x2 has not been run here, and a score above 0.50 is not
established. `MODE='smoke'` is an optional small
environment check; its `smoke_submission.csv` is not for competition submission.
