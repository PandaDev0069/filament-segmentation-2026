"""Regression checks for repetition semantics, source safety, and failure handling."""
import ast
import copy
import json
from pathlib import Path
import sys
import threading

import cv2
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import (DEFAULTS, apply_filter, image_directory, load_config, read_gray,
                    run_experiment, select_images, validate_config, write_png)
from solar import apply_solar


@pytest.fixture
def sample(tmp_path):
    dataset = tmp_path / "MAGFilo" / "test" / "test_images"
    dataset.mkdir(parents=True)
    rng = np.random.default_rng(8)
    original = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    for index in range(12):
        write_png(dataset / f"{index:02d}_太陽.png", original)
    cfg = validate_config({"input_dir": str(dataset.parents[1]), "output_dir": str(tmp_path / "out"),
                           "count": 3, "repeats": [1, 2, 5], "filters": ["median", "bilateral", "clahe"],
                           "thumbnail_size": 64, "opencv_threads": 1})
    return cfg, dataset, original


def test_root_resolution_selection_and_seed(sample):
    cfg, folder, _ = sample
    assert image_directory(cfg["input_dir"]) == folder
    assert image_directory(folder.parent) == folder
    assert image_directory(folder) == folder
    assert [p.name[:2] for p in select_images(cfg)] == ["00", "05", "11"]
    cfg.update(selection="random", seed=13)
    first = select_images(cfg)
    assert first == select_images(cfg)
    cfg["seed"] = 14
    assert first != select_images(cfg)
    cfg["count"] = 13
    with pytest.raises(ValueError, match="only 12"):
        select_images(cfg)


def test_sweep_is_cumulative_and_filters_start_from_original(sample):
    cfg, folder, original = sample
    source = sorted(folder.iterdir())[0]
    before = source.read_bytes()
    out, manifest = run_experiment(cfg, [source])
    assert manifest["status"] == "completed"
    assert len(manifest["records"]) == 10
    for rec in manifest["records"]:
        expected = original.copy()
        for _ in range(rec["passes"]):
            expected = apply_filter(expected, rec["filter"], cfg)
        actual = read_gray(out / rec["file"])
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == original.shape
        assert (out / rec["thumbnail"]).is_file()
    assert source.read_bytes() == before
    assert (out / "metrics.csv").is_file()
    assert (out / "index.html").is_file()
    assert (out / manifest["comparisons"][0]["sheet"]).is_file()
    assert load_config(out / "config.json") == cfg
    # Every new sweep gets its own folder; an old sweep cannot be overwritten.
    another, _ = run_experiment({**cfg, "filters": ["none"]}, [source])
    assert another != out


@pytest.mark.parametrize("patch,match", [
    ({"repeats": []}, "nonempty"),
    ({"repeats": [0]}, "repeats"),
    ({"repeats": [1.5]}, "integer"),
    ({"repeats": [True]}, "integer"),
    ({"median": {"kernel_size": 4}}, "odd"),
    ({"bilateral": {"sigma_color": float("nan")}}, "finite"),
    ({"solar": {"radial_gain_limits": [2, 1]}}, "ordered"),
    ({"solar": {"blackhat_kernels": [3, 5]}}, "3 numbers"),
    ({"filters": ["unknown"]}, "filters"),
    ({"clahe": {"clip_limt": 3}}, "Unknown"),
    ({"repetitions": [1]}, "Unknown"),
])
def test_invalid_parameters_are_rejected(patch, match):
    with pytest.raises(ValueError, match=match):
        validate_config(patch)


def test_repeats_are_sorted_and_deduplicated():
    assert validate_config({"repeats": [5, 1, 5, 2]})["repeats"] == [1, 2, 5]


def test_source_folder_cannot_be_output(sample):
    cfg, folder, _ = sample
    cfg["output_dir"] = str(folder / "filtered")
    with pytest.raises(ValueError, match="outside"):
        run_experiment(cfg)
    assert not (folder / "filtered").exists()


def test_corrupt_image_does_not_stop_other_images(sample):
    cfg, folder, _ = sample
    broken = folder / "broken.png"
    broken.write_bytes(b"not an image")
    cfg["filters"] = ["none"]
    out, manifest = run_experiment(cfg, [broken, sorted(folder.iterdir())[0]])
    assert manifest["status"] == "completed_with_errors"
    assert len(manifest["failures"]) == 1
    assert len(manifest["comparisons"]) == 1
    assert manifest["comparisons"][0]["complete"]
    assert "Errors / skipped" in (out / "index.html").read_text(encoding="utf-8")


def test_cancel_writes_inspectable_partial_results(sample):
    cfg, folder, _ = sample
    cancel = threading.Event()
    def progress(done, total, message):
        if done >= 2:
            cancel.set()
    out, manifest = run_experiment(cfg, [sorted(folder.iterdir())[0]], progress, cancel)
    assert manifest["status"] == "cancelled"
    assert len(manifest["records"]) == 2
    assert not manifest["comparisons"][0]["complete"]
    assert (out / "index.html").is_file()
    assert json.loads((out / "manifest.json").read_text())["status"] == "cancelled"


def test_inputs_not_mutated_and_blend_can_be_disabled(sample):
    cfg, _, original = sample
    before = original.copy()
    for name in cfg["filters"]:
        apply_filter(original, name, cfg)
        np.testing.assert_array_equal(original, before)
    cfg["clahe"]["strength"] = 0
    np.testing.assert_array_equal(apply_filter(original, "clahe", cfg), original)
    for value in (0, 128, 255):
        flat = np.full((128, 128), value, dtype=np.uint8)
        np.testing.assert_array_equal(apply_solar(flat, cfg["solar"]), flat)


def test_reading_unicode_paths_color_and_explicit_bit_depth(sample):
    _, folder, original = sample
    np.testing.assert_array_equal(read_gray(sorted(folder.iterdir())[0]), original)
    color = folder / "rgb.png"
    Image.new("RGB", (10, 12), (255, 0, 0)).save(color)
    result = read_gray(color)
    assert result.shape == (12, 10) and np.all(result == 76)
    depth16 = folder / "16bit.png"
    Image.fromarray(np.full((12, 10), 1000, dtype=np.uint16)).save(depth16)
    with pytest.raises(ValueError, match="8-bit"):
        read_gray(depth16)


def test_solar_matches_existing_notebook_transform():
    notebook = Path(__file__).resolve().parents[2] / "Best" / "solar-filament-yolo26-full-res-0-36-preprocessing.ipynb"
    if not notebook.exists():
        pytest.skip("Reference notebook is not distributed with this standalone app")
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"] if "def estimate_disk(image):" in "".join(c.get("source", [])))
    # Evaluate only function definitions, never notebook setup or training cells.
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    cfg = copy.deepcopy(DEFAULTS["solar"])
    namespace = {"cv2": cv2, "np": np, "PREPROCESSING_CONFIG": cfg}
    exec(compile(tree, str(notebook), "exec"), namespace)
    yy, xx = np.ogrid[:512, :512]
    radius = np.sqrt((yy - 255.5) ** 2 + (xx - 255.5) ** 2)
    image = np.where(radius < 230, 180 - 35 * (radius / 230) ** 2, 0).astype(np.uint8)
    image[200:208, 150:350] = 80
    for _ in range(2):
        expected = namespace["solar_preprocess"](image, cfg)
        actual = apply_solar(image, cfg)
        np.testing.assert_array_equal(actual, expected)
        image = expected
