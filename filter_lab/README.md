# MAGFilo Filter Lab

A standalone Python desktop app for comparing filters on 10 MAGFilo test images.
Everything lives in this folder; source images are read only. Each sweep gets a new
output folder. No models, training, GPU, or web service are needed.

## Open the app

Double-click **`run_gui.bat`**. The launcher uses this project's `.venv` when it
exists, otherwise the `python` command on your PATH.

From PowerShell at the project root:

```powershell
.\.venv\Scripts\python.exe .\filter_lab\filter_lab.py
```

If running in a different Python environment, install the three dependencies first:

```powershell
python -m pip install -r .\filter_lab\requirements.txt
```

The desktop interface uses Tkinter, included with the standard Windows Python
installer. If Tkinter is absent, install Python's Tcl/Tk component. On Linux it is
usually supplied by the `python3-tk` package.

## Use it

1. Leave the input set to `D:\filament-segmentation-2026\MAGFilo_1.0_Kaggle_2026`.
   The app locates `test\test_images` automatically. The `test` folder or the image
   folder itself also works.
2. Choose the image count and selection method, then **Refresh images**. The
   default selects 10 images spread across the sorted filenames. `first` takes the
   first 10; `random` uses the editable seed for repeatable sampling.
3. Edit filter parameters and the comma-separated pass counts: `1,2,3,5,10`, for
   example. Scroll the left settings pane to reach all solar settings.
4. Select an image and click **Preview selected**, or use **Export all selected**
   for the complete sweep. Previews also save full-resolution outputs for that
   one image. Filtering runs in a worker thread; **Cancel** takes effect after the
   current pass and retains completed results.
5. Scroll the comparison and click a tile for an enlarged view. Select **100%** or
   **200%**, drag to pan, and toggle **Show original** to compare at the same
   position. **Open gallery** opens all saved comparisons in your browser.
6. **Save JSON** stores reusable settings. **Load JSON** restores them.

Settings are captured when a run starts. Edits made during a run apply to the next
run. The preview caption identifies the saved run currently on screen.

## Filters and repetition

| Filter | What changes | Main settings |
| --- | --- | --- |
| No filter | Decoded grayscale pixels stay identical | Always included as the baseline |
| Median | Median of a local neighborhood | Odd kernel size: 3, 5, 7, ... |
| Bilateral | Smooths using distance and intensity similarity | Odd diameter, sigma color, sigma space |
| CLAHE | Local contrast equalization | Clip limit, tile grid, blend strength |
| Solar | Disk-aware radial correction, intensity normalization, and dark-filament enhancement | Radial strength/gain limits, target mean/std, morphology strength, disk and salience settings |

“CHALE” is interpreted as **CLAHE**. “Solar” uses the solar preprocessing already
in `Best/solar-filament-yolo26-full-res-0-36-preprocessing.ipynb`, extracted into
`solar.py` without executing or importing that notebook. It is not solarization.
The solar transform preserves the limb/background with a soft blend. If disk
fitting is unreliable, it returns that pass's input unchanged. Comparison labels,
the detail viewer, the gallery, and the manifest identify this fallback for saved
passes.

Each filter starts from the **original image**, independently of the others:

```text
original -> median -> median -> median -> ...
original -> bilateral -> bilateral -> bilateral -> ...
original -> CLAHE -> CLAHE -> CLAHE -> ...
original -> solar -> solar -> solar -> ...
```

`x5` is the fifth successive application of the **same** filter. The program
calculates passes 1 through the largest requested count once and saves only the
requested milestones. Add any positive counts up to 1000. Original is stored once
per image as `none_x000.png` and shown beside each filter row for comparison.

All filters run on the **native 2048 x 2048 images**. Thumbnails are created only
after filtering. PNG outputs are lossless; the no-filter baseline preserves decoded
pixels rather than the original JPEG encoding. Color inputs are converted to
grayscale. Non-8-bit inputs are rejected explicitly to avoid hidden intensity
rescaling. Repeated CLAHE or solar processing can become aggressive; that behavior
is intentionally visible in the comparison.

## Batch mode and editable JSON

Double-click **`run_batch.bat`** to run the settings in `config.json`, or use:

```powershell
.\.venv\Scripts\python.exe .\filter_lab\filter_lab.py --batch --open

# Different pass counts and sample selection:
.\.venv\Scripts\python.exe .\filter_lab\filter_lab.py --batch --count 10 --selection random --seed 123 --repeats 1 2 3 5 10 20

# Only median and bilateral (the original is still included):
.\.venv\Scripts\python.exe .\filter_lab\filter_lab.py --batch --filters median bilateral

# Reuse settings saved from the app or an earlier run:
.\.venv\Scripts\python.exe .\filter_lab\filter_lab.py --batch --config .\filter_lab\my_settings.json
```

Edit `config.json` directly to change defaults. `repeats`, `count`, `selection`,
`filters`, and each filter's parameter group are the main controls. Relative paths
inside a JSON file are resolved relative to that JSON file. CLI `--input` and
`--output` paths are resolved relative to your shell's working directory. The GUI
saves absolute paths in its settings files and all run snapshots.

Other CLI overrides are `--input`, `--output`, `--seed`, and `--selection`. Use
`--help` for the complete list. CLI exit codes: 0 success, 1 image failures,
2 invalid settings or filesystem error, 130 cancelled. Corrupt images are reported
and skipped, allowing other selected images to finish. Invalid values and unknown
settings are rejected before work begins.

## Outputs

The default output parent is `filter_lab\outputs`. Each `run_...` or `preview_...`
folder contains:

```text
index.html                 # Browser gallery with links to native PNGs
config.json                # Exact settings, ready to rerun
manifest.json              # Selected sources, results, failures, versions, status
metrics.csv                # Cumulative filter time, mean/std, mean absolute change
01_<source filename>/
    none_x000.png          # Native original grayscale pixels
    median_x001.png
    median_x002.png
    ...
    *_thumb.jpg            # Small browser thumbnails
    comparison.jpg         # Labeled grid for this source
```

With 10 images, four active filters, and five pass counts, a completed run saves
**210 native PNGs**: 10 originals plus 200 filtered results, alongside 210 thumbnails
and 10 comparison sheets. Output is forbidden inside the source-image directory.
Only one source image and its current filter result need be processed at a time.
Native sweeps can use significant disk space; thumbnails do not replace the PNGs.

`filter_ms` is cumulative processing time through the reported pass, excluding
PNG export time. Measurements describe pixel changes, not segmentation accuracy.
No labels or segmentation models are used.

## Verification

The focused tests check cumulative filtering, independent starts, pixel-identical
baselines, Unicode paths, reproducible selection, cancellation, corrupt-image
handling, safe output paths, parameter validation, and equivalence of the solar
transform to the existing notebook:

```powershell
.\.venv\Scripts\python.exe -m pytest .\filter_lab\tests -q
```

`pytest` is needed only for these development checks, not for running the app.
