"""Standalone solar transform from Best's preprocessing notebook.

Adapted from Best/solar-filament-yolo26-full-res-0-36-preprocessing.ipynb.
No training code, notebook execution, or model dependencies are required.
The disk is estimated afresh on each pass, as in the original transform.
"""
from __future__ import annotations

import cv2
import numpy as np


def estimate_disk(image: np.ndarray) -> dict | None:
    h, w = image.shape
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 20 or cv2.contourArea(contour) < 0.15 * small.size:
        return None
    (cx, cy), (a, b), _ = cv2.fitEllipse(contour)
    radius = (a + b) / 4
    points = contour[:, 0, :].astype(np.float32)
    residual = np.percentile(np.abs(np.linalg.norm(points - [cx, cy], axis=1) - radius), 90)
    if not (64 < radius < 132 and abs(cx - 127.5) < 32 and abs(cy - 127.5) < 32
            and min(a, b) / max(a, b) > 0.90 and residual / radius < 0.06
            and points.min() > 0 and points.max() < 255):
        return None
    return {"cx": float(cx * w / 256), "cy": float(cy * h / 256),
            "radius": float(radius * min(h, w) / 256)}


def radius_grid(shape: tuple, disk: dict) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return np.sqrt((xx - disk["cx"]) ** 2 + (yy - disk["cy"]) ** 2).astype(np.float32) / disk["radius"]


def radial_correct(image: np.ndarray, disk: dict, cfg: dict) -> np.ndarray:
    rr = radius_grid(image.shape, disk)
    small = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    scaled = {"cx": disk["cx"] * 512 / image.shape[1],
              "cy": disk["cy"] * 512 / image.shape[0],
              "radius": disk["radius"] * 512 / min(image.shape)}
    rsmall = radius_grid(small.shape, scaled)
    centers = (np.arange(64) + 0.5) / 64
    profile = np.full(64, np.nan, dtype=np.float32)
    for i in range(64):
        pixels = small[(rsmall >= i / 64) & (rsmall < (i + 1) / 64)]
        if pixels.size >= 16:
            profile[i] = np.median(pixels)
    valid = np.isfinite(profile) & (profile > 5)
    if valid.sum() < 32:
        return image.copy()
    profile = np.interp(centers, centers[valid], profile[valid]).astype(np.float32)
    profile = cv2.GaussianBlur(profile[None, :], (9, 1), 0).ravel()
    target = float(np.median(profile[centers < 0.65]))
    background = np.interp(rr.ravel(), centers, profile).reshape(image.shape)
    gain = np.clip(target / np.maximum(background, 10), *cfg["radial_gain_limits"])
    taper = np.clip((1.02 - rr) / 0.12, 0, 1)
    result = image.astype(np.float32) * (1 + cfg["radial_strength"] * taper * (gain - 1))
    return np.ascontiguousarray(np.clip(np.rint(result), 0, 255).astype(np.uint8))


def morphology_salience(image: np.ndarray, disk: dict, cfg: dict) -> np.ndarray:
    size = cfg["salience_size"]
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    scaled = {"cx": disk["cx"] * size / image.shape[1],
              "cy": disk["cy"] * size / image.shape[0],
              "radius": disk["radius"] * size / min(image.shape)}
    inside = radius_grid(small.shape, scaled) <= cfg["disk_stats_radius"]
    if inside.sum() < 100:
        return np.zeros(image.shape, np.uint8)
    gray = small.astype(np.float32)
    scale = size / 1024
    background = (0.65 * cv2.GaussianBlur(gray, (0, 0), 28 * scale)
                  + 0.35 * cv2.GaussianBlur(gray, (0, 0), 14 * scale))
    salience = 0.50 * np.clip((background - gray) / np.maximum(background, 25), 0, 1)
    for weight, width in zip([0.15, 0.20, 0.15], cfg["blackhat_kernels"]):
        width = max(3, int(round(width * scale)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        salience += weight * cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, kernel).astype(np.float32) / 255
    values = salience[inside]
    median = float(np.median(values))
    mad = max(1.4826 * float(np.median(np.abs(values - median))), 1e-5)
    z = np.clip((salience - median) / mad, 0, 255 / 32)
    z = cv2.resize(z, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    z[radius_grid(image.shape, disk) > cfg["disk_margin"]] = 0
    return np.ascontiguousarray(np.rint(z * 32).astype(np.uint8))


def apply_solar(image: np.ndarray, cfg: dict) -> np.ndarray:
    disk = estimate_disk(image)
    if disk is None:
        return image.copy()
    corrected = radial_correct(image, disk, cfg).astype(np.float32)
    rr = radius_grid(image.shape, disk)
    values = corrected[rr <= cfg["disk_stats_radius"]]
    if values.size == 0 or float(values.std()) < 1.0:
        return image.copy()
    normalized = cfg["normalized_mean"] + cfg["normalized_std"] * (corrected - float(values.mean())) / float(values.std())
    z = morphology_salience(image, disk, cfg).astype(np.float32) / 32
    normalized -= cfg["morphology_strength"] * 8 * np.minimum(z, 6)
    blend = np.clip((cfg["disk_margin"] - rr) / 0.04, 0, 1)
    result = normalized * blend + image.astype(np.float32) * (1 - blend)
    return np.ascontiguousarray(np.clip(np.rint(result), 0, 255).astype(np.uint8))
