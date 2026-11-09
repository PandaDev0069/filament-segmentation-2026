"""Validated configuration, independent filter sweeps, and reproducible exports."""
from __future__ import annotations

import copy
import csv
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Callable
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from solar import apply_solar, estimate_disk

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HERE / "config.json"
FILTER_NAMES = {"none": "No filter", "median": "Median", "bilateral": "Bilateral",
                "clahe": "CLAHE", "solar": "Solar"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULTS = {
    "input_dir": "../MAGFilo_1.0_Kaggle_2026", "output_dir": "outputs", "count": 10,
    "selection": "evenly", "seed": 42, "repeats": [1, 2, 3, 5, 10],
    "filters": list(FILTER_NAMES), "thumbnail_size": 220, "opencv_threads": 4,
    "median": {"kernel_size": 3},
    "bilateral": {"diameter": 9, "sigma_color": 50.0, "sigma_space": 50.0},
    "clahe": {"clip_limit": 2.0, "tile_grid": [8, 8], "strength": 1.0},
    "solar": {"radial_strength": 0.65, "radial_gain_limits": [0.8, 1.25],
              "disk_margin": 1.02, "disk_stats_radius": 0.95, "normalized_mean": 128.0,
              "normalized_std": 32.0, "morphology_strength": 0.25, "salience_size": 1024,
              "blackhat_kernels": [9, 19, 33]},
}


class Cancelled(Exception):
    """The user cancelled between filter passes."""


def _merge(defaults: dict, supplied: dict, prefix: str = "") -> dict:
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'Configuration'} must be a JSON object.")
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown setting: {prefix}{sorted(unknown)[0]}")
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        result[key] = _merge(defaults[key], value, prefix + key + ".") if isinstance(defaults[key], dict) else value
    return result


def _number(value, name: str, low: float, high: float, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}.")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low:g} and {high:g}.")


def _pair(value, name: str, low: float, high: float, integer: bool = False, length: int = 2):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers.")
    for v in value:
        _number(v, name, low, high, integer)


def validate_config(supplied: dict, base: Path = HERE) -> dict:
    cfg = _merge(DEFAULTS, supplied)
    for key in ("input_dir", "output_dir"):
        if not isinstance(cfg[key], str) or not cfg[key].strip():
            raise ValueError(f"{key} must be a nonempty folder path.")
        path = Path(cfg[key]).expanduser()
        cfg[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    _number(cfg["count"], "count", 1, 100000, True)
    _number(cfg["seed"], "seed", 0, 2**32 - 1, True)
    _number(cfg["thumbnail_size"], "thumbnail_size", 64, 512, True)
    _number(cfg["opencv_threads"], "opencv_threads", 1, 64, True)
    if cfg["selection"] not in ("evenly", "first", "random"):
        raise ValueError("selection must be evenly, first, or random.")
    if not isinstance(cfg["repeats"], list) or not cfg["repeats"]:
        raise ValueError("repeats must be a nonempty list, for example [1, 2, 3, 5, 10].")
    for n in cfg["repeats"]:
        _number(n, "repeats", 1, 1000, True)
    cfg["repeats"] = sorted(set(cfg["repeats"]))
    if not isinstance(cfg["filters"], list) or not cfg["filters"] or any(not isinstance(f, str) or f not in FILTER_NAMES for f in cfg["filters"]):
        raise ValueError("filters must be a list of: none, median, bilateral, clahe, solar.")
    cfg["filters"] = [f for f in FILTER_NAMES if f == "none" or f in cfg["filters"]]
    _number(cfg["median"]["kernel_size"], "median.kernel_size", 3, 255, True)
    _number(cfg["bilateral"]["diameter"], "bilateral.diameter", 1, 255, True)
    for section, key in (("median", "kernel_size"), ("bilateral", "diameter")):
        if cfg[section][key] % 2 != 1:
            raise ValueError(f"{section}.{key} must be odd (3, 5, 7, ...).")
    for key in ("sigma_color", "sigma_space"):
        _number(cfg["bilateral"][key], "bilateral." + key, 0.001, 10000)
    _number(cfg["clahe"]["clip_limit"], "clahe.clip_limit", 0.001, 256)
    _pair(cfg["clahe"]["tile_grid"], "clahe.tile_grid", 1, 128, True)
    _number(cfg["clahe"]["strength"], "clahe.strength", 0, 1)
    solar = cfg["solar"]
    for key in ("radial_strength", "morphology_strength"):
        _number(solar[key], "solar." + key, 0, 1)
    _pair(solar["radial_gain_limits"], "solar.radial_gain_limits", 0.01, 10)
    if solar["radial_gain_limits"][0] > solar["radial_gain_limits"][1]:
        raise ValueError("solar.radial_gain_limits must be ordered low, high.")
    for key, low, high in (("disk_margin", 0.9, 1.2), ("disk_stats_radius", 0.1, 1),
                           ("normalized_mean", 0, 255), ("normalized_std", 0.1, 128)):
        _number(solar[key], "solar." + key, low, high)
    _number(solar["salience_size"], "solar.salience_size", 64, 4096, True)
    _pair(solar["blackhat_kernels"], "solar.blackhat_kernels", 3, 255, True, 3)
    if any(n % 2 != 1 for n in solar["blackhat_kernels"]):
        raise ValueError("solar.blackhat_kernels must be three odd sizes.")
    return cfg


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        return validate_config(json.load(stream), path.parent)


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def image_directory(path: str | Path) -> Path:
    """Accept the dataset root, test folder, or test_images folder."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")
    for candidate in (root / "test" / "test_images", root / "test_images", root):
        if candidate.is_dir() and any(p.is_file() and p.suffix.lower() in EXTENSIONS for p in candidate.iterdir()):
            return candidate
    raise ValueError(f"No supported images found in {root} or its test/test_images folder.")


def select_images(cfg: dict) -> list[Path]:
    folder = image_directory(cfg["input_dir"])
    paths = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS), key=lambda p: p.name.lower())
    count = cfg["count"]
    if count > len(paths):
        raise ValueError(f"Requested {count} images, but only {len(paths)} are available in {folder}.")
    if cfg["selection"] == "first":
        return paths[:count]
    if cfg["selection"] == "random":
        return sorted(random.Random(cfg["seed"]).sample(paths, count))
    return [paths[int(i)] for i in np.linspace(0, len(paths) - 1, count, dtype=int)]


def read_gray(path: Path) -> np.ndarray:
    # imdecode/fromfile supports Windows paths containing non-ASCII characters.
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"Cannot read image {path}: {exc}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected 8-bit image; got {image.dtype}: {path.name}. Convert it explicitly before comparing filters.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if image.ndim != 2 or min(image.shape) < 3:
        raise ValueError(f"Expected a grayscale image at least 3 x 3 pixels: {path.name}")
    return np.ascontiguousarray(image)


def apply_filter(image: np.ndarray, name: str, cfg: dict) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 2:
        raise ValueError("Filters require a 2-D uint8 image.")
    if name == "none":
        return image.copy()
    if name == "median":
        return cv2.medianBlur(image, cfg["median"]["kernel_size"])
    if name == "bilateral":
        params = cfg["bilateral"]
        return cv2.bilateralFilter(image, params["diameter"], params["sigma_color"], params["sigma_space"], borderType=cv2.BORDER_REFLECT_101)
    if name == "clahe":
        params = cfg["clahe"]
        enhanced = cv2.createCLAHE(clipLimit=params["clip_limit"], tileGridSize=tuple(params["tile_grid"])).apply(image)
        return cv2.addWeighted(image, 1 - params["strength"], enhanced, params["strength"], 0)
    if name == "solar":
        return apply_solar(image, cfg["solar"])
    raise ValueError(f"Unknown filter: {name}")


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"Cannot encode output: {path}")
    temporary = path.with_suffix(".png.tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded.tobytes())
    temporary.replace(path)


def _font(size: int):
    for name in ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheet(folder: Path, records: list[dict], cfg: dict, title: str) -> Path:
    size = cfg["thumbnail_size"]
    tile_h, gap, top = size + 38, 10, 50
    filters = [f for f in cfg["filters"] if f != "none"] or ["none"]
    columns = [0] + cfg["repeats"] if filters != ["none"] else [0]
    sheet = Image.new("RGB", (gap + len(columns) * (size + gap), top + len(filters) * (tile_h + gap)), "#101820")
    draw = ImageDraw.Draw(sheet)
    title_font = _font(17)
    while len(title) > 3 and draw.textlength(title, font=title_font) > sheet.width - 2 * gap:
        title = title[:-4] + "..."
    draw.text((gap, 12), title, fill="white", font=title_font)
    by_key = {(r["filter"], r["passes"]): r for r in records}
    for row, name in enumerate(filters):
        for col, passes in enumerate(columns):
            key = ("none", 0) if passes == 0 else (name, passes)
            x, y = gap + col * (size + gap), top + row * (tile_h + gap)
            label = "Original / no filter" if passes == 0 else f"{FILTER_NAMES[name]}  x{passes}"
            if key in by_key and by_key[key]["note"]:
                label += " (no disk)"
            draw.text((x, y), label, fill="#d9e4ef", font=_font(14))
            if key in by_key:
                with Image.open(folder / Path(by_key[key]["file"]).name) as full:
                    thumb = full.convert("RGB")
                    thumb.thumbnail((size, size), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x + (size - thumb.width) // 2, y + 27 + (size - thumb.height) // 2))
    path = folder / "comparison.jpg"
    sheet.save(path, quality=94)
    return path


def _gallery(run_dir: Path, manifest: dict) -> None:
    cfg = manifest["config"]
    parts = ["""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MAGFilo filter comparison</title>
<style>body{font:16px system-ui;background:#101820;color:#e6edf4;margin:24px}a{color:#89c7ff}h1{font-size:26px}
.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(160px,1fr));gap:10px;min-width:1000px}.scroll{overflow:auto}
figure{margin:0;background:#1a2936;padding:8px;border-radius:8px}img{width:100%;display:block;image-rendering:auto}
figcaption{padding-bottom:8px}small{color:#b1c4d6}section{margin:32px 0}details{margin:16px 0}pre{white-space:pre-wrap}</style>
<h1>MAGFilo filter comparison</h1>
<p>Every filter starts from the same original. xN means N successive applications of that filter.
Images were processed at native resolution. Click a tile for its full-resolution PNG.</p>"""]
    completed = sum(entry["complete"] for entry in manifest["comparisons"])
    parts.append(f"<p>Status: <b>{html.escape(manifest['status'])}</b>. Requested: {len(manifest['selected_images'])}; completed: {completed}; failed: {len(manifest['failures'])}.</p>")
    parts.append('<p><a href="config.json">Settings</a> &middot; <a href="manifest.json">Manifest</a> &middot; <a href="metrics.csv">Measurements CSV</a></p>')
    parts.append("<p><small>Measurements describe pixel changes and elapsed filter time; they do not measure segmentation accuracy. The no-filter PNG preserves the decoded grayscale pixels.</small></p>")
    for entry in manifest["comparisons"]:
        source = entry["source"]
        records = [r for r in manifest["records"] if r["source"] == source]
        lookup = {(r["filter"], r["passes"]): r for r in records}
        methods = [f for f in cfg["filters"] if f != "none"] or ["none"]
        parts.append(f"<section><h2>{html.escape(Path(source).name)}</h2><p><a href=\"{quote(entry['sheet'])}\">Comparison sheet</a></p><div class=scroll>")
        for name in methods:
            cols = [0] + cfg["repeats"] if name != "none" else [0]
            parts.append(f'<div class="grid" style="--cols:{len(cols)}">')
            for passes in cols:
                rec = lookup.get(("none", 0) if passes == 0 else (name, passes))
                if rec is None:
                    parts.append("<figure><figcaption>Incomplete</figcaption></figure>")
                    continue
                label = "Original / no filter" if passes == 0 else f"{FILTER_NAMES[name]} x{passes}"
                href = quote(rec["file"])
                # Browser thumbnails are separate files; a page never decodes all native PNGs at once.
                thumb = quote(rec["thumbnail"])
                note = f'<br><small>{html.escape(rec["note"])}</small>' if rec["note"] else ""
                parts.append(f'<figure><figcaption>{label}<br><small>{rec["filter_ms"]:.1f} ms cumulative</small>{note}</figcaption><a href="{href}" target="_blank"><img loading="lazy" src="{thumb}" alt="{label}"></a></figure>')
            parts.append("</div><br>")
        parts.append("</div></section>")
    if manifest["failures"]:
        parts.append("<h2>Errors / skipped images</h2><pre>" + html.escape(json.dumps(manifest["failures"], indent=2)) + "</pre>")
    parts.append("<details><summary>Exact settings</summary><pre>" + html.escape(json.dumps(cfg, indent=2)) + "</pre></details></html>")
    (run_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def run_experiment(cfg: dict, paths: list[Path] | None = None,
                   progress: Callable[[int, int, str], None] | None = None,
                   cancel: threading.Event | None = None, prefix: str = "run") -> tuple[Path, dict]:
    cfg = validate_config(cfg)
    paths = select_images(cfg) if paths is None else [Path(p).resolve() for p in paths]
    if not paths:
        raise ValueError("No images selected.")
    input_dir = image_directory(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"]).resolve()
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("Choose an output folder outside test_images to keep the dataset clean.")
    if any(output_dir == p.parent or p.parent in output_dir.parents for p in paths):
        raise ValueError("Output must be outside every selected source image folder.")
    cv2.setNumThreads(cfg["opencv_threads"])
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / (prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir()
    save_json(run_dir / "config.json", cfg)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running", "config": cfg, "resolved_input_dir": str(input_dir),
                "software": {"opencv": cv2.__version__, "numpy": np.__version__, "pillow": Image.__version__},
                "selected_images": [str(p) for p in paths], "records": [], "comparisons": [], "failures": []}
    active = [f for f in cfg["filters"] if f != "none"]
    total = len(paths) * (1 + len(active) * max(cfg["repeats"]))
    done = 0
    started = time.perf_counter()

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    def tick(message: str):
        if progress:
            progress(done, total, message)

    def snapshot(folder: Path, original: np.ndarray, result: np.ndarray, source: Path,
                 method: str, passes: int, seconds: float, note: str = "") -> dict:
        file = folder / f"{method}_x{passes:03d}.png"
        write_png(file, result)
        thumbnail = file.with_name(file.stem + "_thumb.jpg")
        thumb = Image.fromarray(result)
        thumb.thumbnail((cfg["thumbnail_size"], cfg["thumbnail_size"]), Image.Resampling.LANCZOS)
        thumb.save(thumbnail, quality=92)
        record = {"source": str(source), "filter": method, "passes": passes,
                  "width": result.shape[1], "height": result.shape[0], "filter_ms": round(seconds * 1000, 3),
                  "mean": round(float(result.mean()), 4), "std": round(float(result.std()), 4),
                  "mean_absolute_change": round(float(cv2.absdiff(original, result).mean()), 4),
                  "file": file.relative_to(run_dir).as_posix(),
                  "thumbnail": thumbnail.relative_to(run_dir).as_posix(), "note": note}
        manifest["records"].append(record)
        return record

    try:
        for index, source in enumerate(paths, 1):
            check_cancel()
            tick(f"Reading {index}/{len(paths)}: {source.name}")
            folder = run_dir / f"{index:02d}_{source.name}"
            folder.mkdir()
            image_records = []
            image_error = None
            image_done_before = done
            try:
                original = read_gray(source)
                image_records.append(snapshot(folder, original, original, source, "none", 0, 0))
                done += 1
                tick(f"{index}/{len(paths)} | {source.name} | Original")
                for method in active:
                    current = original.copy()
                    cumulative = 0.0
                    for passes in range(1, max(cfg["repeats"]) + 1):
                        check_cancel()
                        before = time.perf_counter()
                        no_disk = method == "solar" and estimate_disk(current) is None
                        current = apply_filter(current, method, cfg)
                        cumulative += time.perf_counter() - before
                        if passes in cfg["repeats"]:
                            note = "No reliable solar disk on this pass: returned input unchanged." if no_disk else ""
                            image_records.append(snapshot(folder, original, current, source, method, passes, cumulative, note))
                        done += 1
                        tick(f"{index}/{len(paths)} | {source.name} | {FILTER_NAMES[method]} x{passes}")
            except (ValueError, cv2.error) as exc:
                image_error = str(exc)
                manifest["failures"].append({"source": str(source), "error": image_error})
                done = image_done_before + 1 + len(active) * max(cfg["repeats"])
                tick(f"Skipped {source.name}: {image_error}")
            finally:
                if image_records:
                    sheet = contact_sheet(folder, image_records, cfg, source.name)
                    manifest["comparisons"].append({"source": str(source), "sheet": sheet.relative_to(run_dir).as_posix(),
                                                     "complete": image_error is None and len(image_records) == 1 + len(active) * len(cfg["repeats"])})
                save_json(run_dir / "manifest.json", manifest)
        manifest["status"] = "completed_with_errors" if manifest["failures"] else "completed"
    except (Cancelled, KeyboardInterrupt):
        manifest["status"] = "cancelled"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failures"].append({"error": str(exc)})
        raise
    finally:
        manifest["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        save_json(run_dir / "manifest.json", manifest)
        fields = ["source", "filter", "passes", "width", "height", "filter_ms", "mean", "std", "mean_absolute_change", "file", "thumbnail", "note"]
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest["records"])
        _gallery(run_dir, manifest)
    return run_dir, manifest
