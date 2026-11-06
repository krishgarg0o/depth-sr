"""Evaluation metrics.

All depth errors are computed on pixels with valid ground truth (gt > 0).
  rmse, mae          : in the depth unit (metres for NYU -> reported in cm)
  psnr               : on depth normalised by the GT max (peak = 1)
  ssim               : on the same normalised depth
  boundary_rmse      : RMSE restricted to a 5 px band around GT depth discontinuities
  edge_f1            : F1 of predicted vs GT depth-discontinuity maps (2 px tolerance)
  texture_copy       : mean |grad pred| in flat-depth regions with strong RGB texture
                       (lower = less texture copying)
"""
from __future__ import annotations

import cv2
import numpy as np

from .classical import depth_discontinuities, rgb_edges


def _valid(gt):
    return gt > 0


def rmse(pred, gt):
    m = _valid(gt)
    return float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)))


def mae(pred, gt):
    m = _valid(gt)
    return float(np.mean(np.abs(pred[m] - gt[m])))


def psnr(pred, gt):
    m = _valid(gt)
    peak = gt[m].max()
    mse = np.mean(((pred[m] - gt[m]) / peak) ** 2)
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def ssim(pred, gt):
    peak = gt.max() + 1e-8
    a = np.clip(pred / peak, 0, 1).astype(np.float64)
    b = np.clip(gt / peak, 0, 1).astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = (11, 11)
    mu_a, mu_b = cv2.GaussianBlur(a, k, 1.5), cv2.GaussianBlur(b, k, 1.5)
    s_aa = cv2.GaussianBlur(a * a, k, 1.5) - mu_a ** 2
    s_bb = cv2.GaussianBlur(b * b, k, 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, k, 1.5) - mu_a * mu_b
    smap = ((2 * mu_a * mu_b + C1) * (2 * s_ab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (s_aa + s_bb + C2))
    return float(smap[_valid(gt)].mean())


def boundary_mask(gt, width: int = 5, rel_thresh: float = 0.05):
    _, disc = depth_discontinuities(gt, rel_thresh)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    return (cv2.dilate(disc, k) > 0) & _valid(gt)


def boundary_rmse(pred, gt, width=5):
    m = boundary_mask(gt, width)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[m] - gt[m]) ** 2)))


def edge_f1(pred, gt, tol: int = 2, rel_thresh: float = 0.05):
    _, ep = depth_discontinuities(np.where(pred > 0, pred, 1e-3), rel_thresh)
    _, eg = depth_discontinuities(gt, rel_thresh)
    ep, eg = ep > 0, eg > 0
    eg &= _valid(gt)
    ep &= _valid(gt)
    if eg.sum() == 0 or ep.sum() == 0:
        return float("nan")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    eg_d = cv2.dilate(eg.astype(np.uint8), k) > 0
    ep_d = cv2.dilate(ep.astype(np.uint8), k) > 0
    prec = (ep & eg_d).sum() / ep.sum()
    rec = (eg & ep_d).sum() / eg.sum()
    return float(2 * prec * rec / (prec + rec + 1e-8))


def texture_copy(pred, gt, rgb):
    """Spurious depth gradient where RGB has texture but GT depth is flat."""
    e_rgb, _ = rgb_edges(rgb)
    gmag, _ = depth_discontinuities(gt, 0.05)
    flat_textured = (e_rgb > 0.3) & (gmag < 0.005) & _valid(gt)
    if flat_textured.sum() < 50:
        return float("nan")
    pmag, _ = depth_discontinuities(np.where(pred > 0, pred, 1e-3), 0.05)
    return float(pmag[flat_textured].mean() * 100)   # in % depth change per pixel


def all_metrics(pred, gt, rgb=None) -> dict:
    out = {
        "RMSE": rmse(pred, gt), "MAE": mae(pred, gt), "PSNR": psnr(pred, gt), "SSIM": ssim(pred, gt),
        "BoundaryRMSE": boundary_rmse(pred, gt), "EdgeF1": edge_f1(pred, gt),
    }
    if rgb is not None:
        out["TextureCopy"] = texture_copy(pred, gt, rgb)
    return out
