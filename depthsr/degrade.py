"""Degradation protocol used to create LR / sparse / noisy depth for training and evaluation.

Documented in docs/design.md so results are reproducible:
  * bicubic downsampling by factor s (standard guided depth SR protocol)
  * optional additive Gaussian noise proportional to depth (ToF/structured-light style)
  * optional random holes (blobs) to mimic missing measurements
  * optional sparse sampling (keep N random points, depth-completion setting)
"""
from __future__ import annotations

import cv2
import numpy as np


def downsample(depth: np.ndarray, scale: int, mode: str = "bicubic") -> np.ndarray:
    h, w = depth.shape
    interp = {"bicubic": cv2.INTER_CUBIC, "nearest": cv2.INTER_NEAREST, "area": cv2.INTER_AREA}[mode]
    lr = cv2.resize(depth, (w // scale, h // scale), interpolation=interp)
    return np.clip(lr, 0, None).astype(np.float32)


def add_noise(depth: np.ndarray, sigma: float = 0.02, rng=None) -> np.ndarray:
    """Depth-dependent Gaussian noise: sigma (relative), e.g. 0.02 -> 2 cm at 1 m."""
    rng = rng or np.random.default_rng()
    noisy = depth + rng.normal(0, 1, depth.shape).astype(np.float32) * sigma * depth
    noisy[depth <= 0] = 0
    return np.clip(noisy, 0, None)


def add_holes(depth: np.ndarray, frac: float = 0.05, rng=None) -> np.ndarray:
    """Zero out random blobs covering ~frac of the image (missing returns)."""
    rng = rng or np.random.default_rng()
    h, w = depth.shape
    mask = np.zeros((h, w), np.uint8)
    target = frac * h * w
    while mask.sum() < target:
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.integers(1, max(2, h // 12)), rng.integers(1, max(2, w // 12))
        cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), float(rng.uniform(0, 180)), 0, 360, 1, -1)
    out = depth.copy()
    out[mask > 0] = 0
    return out


def sparsify(depth: np.ndarray, n_points: int = 500, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    valid = np.flatnonzero(depth > 0)
    keep = rng.choice(valid, size=min(n_points, valid.size), replace=False)
    out = np.zeros_like(depth)
    out.flat[keep] = depth.flat[keep]
    return out


def make_lr(depth_hr: np.ndarray, scale: int, noise: float = 0.0, holes: float = 0.0,
            seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lr = downsample(depth_hr, scale)
    if noise > 0:
        lr = add_noise(lr, noise, rng)
    if holes > 0:
        lr = add_holes(lr, holes, rng)
    return lr
