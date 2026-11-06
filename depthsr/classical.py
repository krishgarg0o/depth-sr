"""Classical computer-vision stage.

Everything here is non-learned and explainable:
  * interpolation baselines (nearest / bilinear / bicubic)
  * hole filling by multi-scale normalised convolution (also densifies sparse depth)
  * adaptive noise estimation -> adaptive filter parameters
  * guided filter and joint bilateral upsampling (JBU)
  * RGB edge detection + depth discontinuity detection
  * "trust map": RGB edges that coincide with depth discontinuities (true boundaries)
    vs RGB edges inside flat depth regions (texture -> must NOT be copied)
  * trust-gated refinement: sharp RGB-guided filtering only near real boundaries
  * boundary snapping to remove "flying pixels" between foreground and background
"""
from __future__ import annotations

import cv2
import numpy as np

HAS_XIMGPROC = hasattr(cv2, "ximgproc")

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
}


# ----------------------------------------------------------------------------- baselines
def interpolate(depth_lr: np.ndarray, out_hw: tuple[int, int], method: str = "bicubic") -> np.ndarray:
    H, W = out_hw
    up = cv2.resize(depth_lr.astype(np.float32), (W, H), interpolation=INTERP[method])
    return np.clip(up, 0, None)


def upsample_with_holes(depth_lr: np.ndarray, out_hw, method="bicubic"):
    """Interpolate while not letting zeros (holes) bleed into valid values."""
    valid = (depth_lr > 0).astype(np.float32)
    if valid.min() > 0:
        return interpolate(depth_lr, out_hw, method), np.ones(out_hw, np.float32)
    num = interpolate(depth_lr * valid, out_hw, method)
    den = interpolate(valid, out_hw, method)
    up = np.where(den > 0.5, num / np.maximum(den, 1e-6), 0).astype(np.float32)
    return up, (den > 0.5).astype(np.float32)


# ----------------------------------------------------------------------------- hole filling
def fill_holes(depth: np.ndarray, max_iter: int = 12) -> np.ndarray:
    """Multi-scale normalised convolution. Works for small holes and very sparse input.

    For each growing Gaussian sigma we compute G*(d*m) / G*m and fill only the pixels
    that are still missing, so valid measurements are never altered.
    """
    d = depth.astype(np.float32).copy()
    m = (d > 0).astype(np.float32)
    if m.all():
        return d
    if not m.any():
        return d
    sigma = 1.0
    for _ in range(max_iter):
        k = int(2 * round(3 * sigma) + 1)
        num = cv2.GaussianBlur(d * m, (k, k), sigma)
        den = cv2.GaussianBlur(m, (k, k), sigma)
        fill = (m == 0) & (den > 1e-3)
        d[fill] = num[fill] / den[fill]
        m[fill] = 1.0
        if m.all():
            break
        sigma *= 2
    if not m.all():   # anything left (huge holes) -> global median
        d[m == 0] = np.median(d[m > 0])
    return d


# ----------------------------------------------------------------------------- adaptivity
def estimate_noise(depth: np.ndarray) -> float:
    """Robust relative noise level: MAD of the Laplacian of log-depth (Immerkaer-style)."""
    valid = depth > 0
    if valid.sum() < 100:
        return 0.0
    ld = np.log(np.where(valid, depth, np.median(depth[valid])))
    lap = cv2.Laplacian(ld, cv2.CV_32F)
    v = lap[valid]
    return float(1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(20))


def adaptive_params(depth_lr: np.ndarray, scale: int) -> dict:
    """Pick filter parameters from image statistics instead of fixed values."""
    noise = estimate_noise(depth_lr)
    valid = depth_lr[depth_lr > 0]
    drange = float(np.percentile(valid, 99) - np.percentile(valid, 1)) if valid.size else 1.0
    return {
        "noise": noise,
        "depth_range": drange,
        # spatial support grows with the upsampling factor
        "radius": int(np.clip(round(scale * 1.5), 2, 10)),
        "sigma_space": float(max(2.0, scale * 1.2)),
        # colour sensitivity: noisier depth -> rely more on colour
        "sigma_color": float(np.clip(12 + 400 * noise, 10, 40)),
        # guided filter regularisation: noisier -> stronger smoothing
        "gf_eps": float(np.clip(1e-4 + 0.5 * noise ** 2 * 1e2, 1e-4, 1e-2)),
        # how much to smooth outside boundaries: 0 for clean input, 1 for clearly noisy input.
        # the estimator's clean-input floor grows ~0.0006 per scale step (LR structure), so subtract it.
        "smooth_w": float(np.clip((noise - 0.004 - 0.0006 * scale) / 0.012, 0, 1)),
    }


# ----------------------------------------------------------------------------- filters
def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al. guided filter with a single-channel guide (float32 in [0,1])."""
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mI, mp = _box(I, r), _box(p, r)
    cov = _box(I * p, r) - mI * mp
    var = _box(I * I, r) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return _box(a, r) * I + _box(b, r)


def joint_bilateral(depth_up: np.ndarray, rgb: np.ndarray, d: int, sigma_color: float,
                    sigma_space: float) -> np.ndarray:
    """Joint bilateral filter of an upsampled depth using the HR RGB as guide (JBU)."""
    if HAS_XIMGPROC:
        guide = rgb.astype(np.float32)
        return cv2.ximgproc.jointBilateralFilter(guide, depth_up.astype(np.float32), d,
                                                 sigma_color, sigma_space)
    # fallback: slower pure-numpy JBU on a window
    return _jbf_numpy(depth_up, rgb, d // 2, sigma_color, sigma_space)


def _jbf_numpy(depth, rgb, r, sc, ss):
    H, W = depth.shape
    pad_d = np.pad(depth, r, mode="reflect")
    pad_g = np.pad(rgb.astype(np.float32), ((r, r), (r, r), (0, 0)), mode="reflect")
    num = np.zeros_like(depth, np.float32)
    den = np.zeros_like(depth, np.float32)
    g0 = rgb.astype(np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ws = np.exp(-(dy * dy + dx * dx) / (2 * ss * ss))
            gs = pad_g[r + dy:r + dy + H, r + dx:r + dx + W]
            wc = np.exp(-((gs - g0) ** 2).sum(-1) / (2 * sc * sc))
            w = ws * wc
            num += w * pad_d[r + dy:r + dy + H, r + dx:r + dx + W]
            den += w
    return num / (den + 1e-8)


# ----------------------------------------------------------------------------- edges
def rgb_edges(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (soft edge magnitude in [0,1], binary Canny edges)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)
    med = np.median(gray)
    canny = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med + 20)))
    return mag, (canny > 0).astype(np.float32)


def depth_discontinuities(depth: np.ndarray, rel_thresh: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of log-depth -> scale invariant discontinuity strength and binary mask.

    A relative jump of `rel_thresh` (5 %) between neighbours counts as a discontinuity.
    """
    valid = depth > 0
    ld = np.log(np.where(valid, depth, np.median(depth[valid]) if valid.any() else 1.0)).astype(np.float32)
    gx = cv2.Sobel(ld, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(ld, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.sqrt(gx * gx + gy * gy)
    return mag, (mag > rel_thresh).astype(np.float32)


def trust_map(rgb: np.ndarray, depth_up: np.ndarray, scale: int):
    """Edge-agreement map.

    band  : soft region (width ~ scale px) around depth discontinuities of the coarse depth.
            The true boundary must lie in here because LR depth blurs it by ~scale px.
    trust : RGB edge strength * band -> RGB edges that are real geometric boundaries.
    texture: RGB edges outside the band -> texture/printed patterns, guidance suppressed.
    """
    e_rgb, _ = rgb_edges(rgb)
    dmag, _ = depth_discontinuities(depth_up, rel_thresh=0.02)
    dsoft = np.clip(dmag / 0.05, 0, 1)
    k = 2 * max(1, scale) + 1
    band = cv2.dilate(dsoft, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    band = cv2.GaussianBlur(band, (0, 0), max(1.0, scale / 2))
    band = np.clip(band, 0, 1).astype(np.float32)
    trust = (e_rgb * band).astype(np.float32)
    texture = (e_rgb * (1 - band)).astype(np.float32)
    return trust, band, texture


def snap_boundaries(depth: np.ndarray, band: np.ndarray, k: int = 5, thresh: float = 0.5) -> np.ndarray:
    """Remove flying pixels: inside strong boundary regions, snap each pixel to the nearer of
    the local foreground (min) or background (max) depth."""
    ker = np.ones((k, k), np.uint8)
    lo = cv2.erode(depth, ker)
    hi = cv2.dilate(depth, ker)
    snapped = np.where(np.abs(depth - lo) < np.abs(depth - hi), lo, hi)
    jump = (hi - lo) / (hi + 1e-6)
    m = (band > thresh) & (jump > 0.1)
    out = depth.copy()
    out[m] = snapped[m]
    return out


# ----------------------------------------------------------------------------- full classical pipeline
def classical_refine(depth_lr: np.ndarray, rgb: np.ndarray, scale: int, return_all: bool = False,
                     use_trust: bool = True, use_fill: bool = True, use_snap: bool = False,
                     use_adapt: bool = True):
    """Classical (non-learned) edge-aware upsampling.

    1. hole fill at LR  2. bicubic up  3. adaptive params (noise estimate)  4. trust map
    5. noise-adaptive base: bicubic for clean input, depth-guided smoothing for noisy input
    6. trust-weighted mix: JBU (sharp, RGB-guided) only where RGB edges agree with depth
       discontinuities; at x16 the whole boundary band is trusted more (LR edges are very coarse)
    7. optional boundary snapping (off by default - it raised RMSE on clean NYU depth)
    """
    H, W = rgb.shape[:2]
    hole_mask_lr = (depth_lr <= 0).astype(np.float32)
    filled_lr = fill_holes(depth_lr) if use_fill else depth_lr
    params = adaptive_params(depth_lr, scale)
    up = interpolate(filled_lr, (H, W), "bicubic")
    if not use_fill:
        up, _ = upsample_with_holes(depth_lr, (H, W))

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    d = 2 * params["radius"] + 1
    import time as _t
    _t0 = _t.perf_counter()
    jbu = joint_bilateral(up, rgb, d, params["sigma_color"], params["sigma_space"])
    jbu_ms = (_t.perf_counter() - _t0) * 1000
    trust, band, texture = trust_map(rgb, up, scale)

    w = params["smooth_w"] if use_adapt else 1.0
    if use_trust:
        # outside boundaries: depth-guided smoothing only as much as the measured noise requires
        dn = up / (up.max() + 1e-6)
        smooth = guided_filter(dn, up, params["radius"], params["gf_eps"])
        base = (1 - w) * up + w * smooth
        t = band * np.clip(2.0 * trust + w + max(0.0, np.log2(scale) - 3), 0, 1)
        refined = t * jbu + (1 - t) * base
    else:
        refined = jbu
    if use_snap:
        refined = snap_boundaries(refined, band)
    refined = np.clip(refined, 0, None).astype(np.float32)

    if not return_all:
        return refined
    hole_up = interpolate(hole_mask_lr, (H, W), "nearest")
    gf_rgb = guided_filter(gray, up, params["radius"], params["gf_eps"])
    return refined, {
        "bicubic": up, "jbu": jbu, "guided_rgb": gf_rgb, "trust": trust, "band": band,
        "texture": texture, "hole_mask": hole_up, "filled_lr": filled_lr, "params": params,
        "jbu_ms": jbu_ms,
    }
