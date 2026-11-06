"""Input loading + validation for RGB and depth maps.

Depth is always handled internally as float32 metres (or relative units for
RGB-only / 8-bit inputs) with 0 meaning "missing".
"""
from __future__ import annotations

import io
import re

import cv2
import numpy as np

MAX_SIDE = 1280          # larger inputs are downscaled to keep latency < 10 s
MIN_SIDE = 16

DEPTH_UNITS = {
    "auto": None,
    "uint16 millimetres (NYU / Middlebury / RealSense)": 1000.0,
    "uint16 KITTI (value / 256 = metres)": 256.0,
    "float metres (.npy / .pfm / .tiff)": 1.0,
    "8-bit relative (0-255)": 25.5,   # mapped to 0-10 'units'
}


class InputError(ValueError):
    """Raised for user-facing input problems (shown in the UI)."""


def read_pfm(buf: bytes) -> np.ndarray:
    f = io.BytesIO(buf)
    header = f.readline().decode().strip()
    if header not in ("Pf", "PF"):
        raise InputError("Not a valid PFM file")
    dims = re.findall(r"\d+", f.readline().decode())
    w, h = int(dims[0]), int(dims[1])
    scale = float(f.readline().decode().strip())
    endian = "<" if scale < 0 else ">"
    data = np.frombuffer(f.read(), dtype=endian + "f4")
    ch = 3 if header == "PF" else 1
    data = data.reshape(h, w, ch)[..., 0] if ch == 3 else data.reshape(h, w)
    data = np.flipud(data).astype(np.float32)
    data[~np.isfinite(data)] = 0
    return data


def load_depth(path: str, unit: str = "auto") -> np.ndarray:
    """Load a depth map from png/tiff/npy/pfm and convert to float32 units."""
    if path is None:
        raise InputError("No depth file given")
    p = path.lower()
    if p.endswith(".npy"):
        d = np.load(path).astype(np.float32)
        raw_dtype = "float"
    elif p.endswith(".pfm"):
        with open(path, "rb") as fh:
            d = read_pfm(fh.read())
        raw_dtype = "float"
    else:
        d = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if d is None:
            raise InputError("Could not decode the depth file. Use PNG/TIFF/NPY/PFM.")
        if d.ndim == 3:
            # colour image given as depth: accept only if it is effectively grey
            if np.abs(d[..., 0].astype(int) - d[..., 1]).mean() > 2:
                raise InputError(
                    "The depth file looks like a colour image, not a depth map. "
                    "Upload a single-channel depth PNG (16-bit preferred).")
            d = d[..., 0]
        raw_dtype = d.dtype.name
        d = d.astype(np.float32)
    if d.ndim != 2:
        raise InputError(f"Depth must be 2-D, got shape {d.shape}")
    d[~np.isfinite(d)] = 0
    d[d < 0] = 0

    if unit == "auto" or unit not in DEPTH_UNITS:
        if raw_dtype == "uint16":
            div = 1000.0
        elif raw_dtype == "uint8":
            div = 25.5
        else:
            div = 1.0
    else:
        div = DEPTH_UNITS[unit]
    d = d / div
    if (d > 0).mean() < 0.001:
        raise InputError("Depth map is empty (no valid pixels).")
    return d


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("Could not decode the RGB image.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_rgb_uint8(img) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img / (img.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
    return img


def limit_size(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = rgb.shape[:2]
    if min(h, w) < MIN_SIDE:
        raise InputError(f"Image too small ({w}x{h}).")
    s = max(h, w)
    if s > max_side:
        f = max_side / s
        rgb = cv2.resize(rgb, (int(w * f) // 8 * 8, int(h * f) // 8 * 8), interpolation=cv2.INTER_AREA)
    return rgb


def check_pair(rgb_hr: np.ndarray, depth_lr: np.ndarray, scale: int) -> np.ndarray:
    """Make sure the LR depth and HR RGB are consistent; returns resized RGB."""
    H, W = rgb_hr.shape[:2]
    h, w = depth_lr.shape
    exp_h, exp_w = h * scale, w * scale
    ar_rgb, ar_d = W / H, w / h
    if abs(ar_rgb - ar_d) / ar_d > 0.05:
        raise InputError(
            f"Aspect ratio mismatch: RGB is {W}x{H}, depth is {w}x{h}. "
            "They must show the same view.")
    if (H, W) != (exp_h, exp_w):
        rgb_hr = cv2.resize(rgb_hr, (exp_w, exp_h), interpolation=cv2.INTER_AREA)
    return rgb_hr


def colorize(depth: np.ndarray, vmin=None, vmax=None, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    d = depth.astype(np.float32)
    valid = d > 0
    if vmin is None:
        vmin = np.percentile(d[valid], 1) if valid.any() else 0
    if vmax is None:
        vmax = np.percentile(d[valid], 99) if valid.any() else 1
    n = np.clip((d - vmin) / (vmax - vmin + 1e-8), 0, 1)
    c = cv2.applyColorMap((n * 255).astype(np.uint8), cmap)
    c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    c[~valid] = 0
    return c


def colorize_map(m: np.ndarray, cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    m = m.astype(np.float32)
    n = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return cv2.cvtColor(cv2.applyColorMap((n * 255).astype(np.uint8), cmap), cv2.COLOR_BGR2RGB)
