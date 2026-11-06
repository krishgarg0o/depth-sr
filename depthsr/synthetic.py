"""Synthetic RGB-D scenes for unit tests and offline sanity checks.

Each scene: a slanted background plane + random foreground shapes at different depths.
RGB = per-object colour + strong *texture that does not exist in depth* (stripes,
checkerboards, printed text) so texture-copying artifacts are measurable.
"""
from __future__ import annotations

import cv2
import numpy as np


def make_scene(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = (3.0 + 2.0 * yy / h + 0.5 * xx / w).astype(np.float32)   # slanted floor/wall
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = rng.integers(60, 200, 3)
    # background texture (checkerboard poster) -> no depth change
    cs = int(rng.integers(6, 14))
    checker = (((yy // cs) + (xx // cs)) % 2).astype(np.float32)
    rgb = np.clip(rgb + (checker[..., None] * 50 - 25), 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(3, 6))):
        mask = np.zeros((h, w), np.uint8)
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        if rng.random() < 0.5:
            r = int(rng.integers(15, h // 4))
            cv2.circle(mask, (cx, cy), r, 1, -1)
        else:
            pts = rng.integers(0, [w, h], size=(int(rng.integers(3, 6)), 2)).astype(np.int32)
            cv2.fillPoly(mask, [cv2.convexHull(pts)], 1)
        m = mask > 0
        z = rng.uniform(0.8, 2.8)
        depth[m] = z + 0.3 * (xx[m] - cx) / w            # slightly slanted object
        col = rng.integers(0, 255, 3)
        rgb[m] = col
        # stripes texture on the object
        if rng.random() < 0.6:
            stripes = ((xx + yy) // int(rng.integers(3, 8)) % 2).astype(bool) & m
            rgb[stripes] = np.clip(col.astype(int) + 60, 0, 255)
    cv2.putText(rgb, "TEXT", (int(w * 0.1), int(h * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    noise = rng.normal(0, 4, rgb.shape)
    rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
    return rgb, depth
