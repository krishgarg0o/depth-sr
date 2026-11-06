"""Hybrid pipeline: classical stage -> TGNet residual refinement -> adaptive fusion.

Methods exposed (used by the app and the evaluation script):
  nearest | bilinear | bicubic            interpolation baselines
  jbu | guided                            classical edge-aware filters
  classical                               full classical pipeline (fill + trust-gated + snap)
  dl                                      TGNet alone (naive upsampled input, no classical maps)
  hybrid                                  classical stage + TGNet + learned fusion gate
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import classical as C
from .model import TGNet

METHODS = ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical", "dl", "hybrid"]
DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "tgnet.pt")


def build_inputs(depth_lr, rgb, scale, mode="hybrid", ablate=None, classical_result=None):
    """Prepare network inputs (numpy) for 'hybrid' or 'dl' mode."""
    ablate = ablate or {}
    H, W = rgb.shape[:2]
    if mode == "hybrid":
        if classical_result is not None and not ablate:
            init, aux = classical_result
            aux = dict(aux)
        else:
            init, aux = C.classical_refine(depth_lr, rgb, scale, return_all=True,
                                           use_trust=not ablate.get("no_trust"),
                                           use_fill=not ablate.get("no_fill"),
                                           use_snap=not ablate.get("no_snap"))
        valid = np.ones((H, W), np.float32) if not ablate.get("no_fill") else (init > 0).astype(np.float32)
        trust, band = aux["trust"], aux["band"]
        if ablate.get("no_trust"):
            trust = np.zeros_like(trust)
            band = np.zeros_like(band)
    else:  # dl only
        init, valid = C.upsample_with_holes(depth_lr, (H, W), "bicubic")
        trust = np.zeros((H, W), np.float32)
        band = np.zeros((H, W), np.float32)
        aux = {}
    v = init[init > 0]
    s = float(np.percentile(v, 99)) if v.size else 1.0
    s = max(s, 1e-3)
    return {
        "init": (init / s).astype(np.float32), "valid": valid.astype(np.float32),
        "scale_plane": np.full((H, W), np.log2(scale) / 4.0, np.float32),
        "rgb": (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1),
        "trust": trust.astype(np.float32), "band": band.astype(np.float32), "norm": s,
    }, aux


class DepthSR:
    def __init__(self, ckpt: str | None = DEFAULT_CKPT, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TGNet().to(self.device).eval()
        self.has_weights = False
        if ckpt and os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location=self.device)
            self.net.load_state_dict(sd.get("model", sd))
            self.has_weights = True
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    @torch.no_grad()
    def _net(self, inp, return_aux=False, no_gate=False):
        def t(x):
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
            if x.dim() == 2:
                x = x[None]
            return x[None].to(self.device)

        H, W = inp["init"].shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        args = [t(inp[k]) for k in ("init", "valid", "scale_plane", "rgb", "trust", "band")]
        if ph or pw:
            args = [F.pad(a, (0, pw, 0, ph), mode="replicate") for a in args]
        pred, aux = self.net(*args, return_aux=True)
        if no_gate:
            pred = args[0] + aux["residual"]
        pred = pred[..., :H, :W]
        out = pred[0, 0].cpu().numpy() * inp["norm"]
        aux = {k: v[0, 0, :H, :W].cpu().numpy() for k, v in aux.items()}
        return np.clip(out, 0, None).astype(np.float32), aux

    def run(self, depth_lr, rgb, scale, method="hybrid", ablate=None, return_aux=False, cache=None):
        """depth_lr: float32 (h,w) ; rgb: uint8 (h*scale, w*scale, 3).

        cache: optional dict shared between calls on the SAME input (used by the app to avoid
        recomputing the classical stage for several methods). Reported latencies stay honest:
        a cached stage adds its originally measured time.
        """
        t0 = time.perf_counter()
        extra_ms = 0.0
        H, W = rgb.shape[:2]
        aux = {}
        if cache is not None and "classical" in cache and method in ("classical", "jbu", "hybrid") and not ablate:
            c_out, c_aux, c_ms = cache["classical"]
            if method == "classical":
                return (c_out, c_aux, c_ms) if return_aux else c_out
            if method == "jbu":
                o = np.clip(c_aux["jbu"], 0, None).astype(np.float32)
                return (o, {}, c_aux["jbu_ms"]) if return_aux else o
            inp, aux = build_inputs(depth_lr, rgb, scale, "hybrid", None, (c_out, c_aux))
            out, net_aux = self._net(inp)
            aux.update(net_aux)
            aux["classical"] = c_out
            ms = (time.perf_counter() - t0) * 1000 + c_ms
            return (out, aux, ms) if return_aux else out
        if method in ("nearest", "bilinear", "bicubic"):
            out, _ = C.upsample_with_holes(depth_lr, (H, W), method)
        elif method == "jbu":
            up = C.interpolate(C.fill_holes(depth_lr), (H, W), "bicubic")
            p = C.adaptive_params(depth_lr, scale)
            out = C.joint_bilateral(up, rgb, 2 * p["radius"] + 1, p["sigma_color"], p["sigma_space"])
        elif method == "guided":
            up = C.interpolate(C.fill_holes(depth_lr), (H, W), "bicubic")
            p = C.adaptive_params(depth_lr, scale)
            import cv2
            g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            out = C.guided_filter(g, up, p["radius"], p["gf_eps"])
        elif method == "classical":
            out, aux = C.classical_refine(depth_lr, rgb, scale, return_all=True)
        elif method in ("dl", "hybrid"):
            if not self.has_weights:
                raise RuntimeError("Model weights not found (checkpoints/tgnet.pt). Train first.")
            inp, aux = build_inputs(depth_lr, rgb, scale, method, ablate)
            out, net_aux = self._net(inp, no_gate=bool(ablate and ablate.get("no_gate")))
            aux.update(net_aux)
            if method == "hybrid":
                aux["classical"] = inp["init"] * inp["norm"]
        else:
            raise ValueError(f"unknown method {method}")
        out = np.clip(out, 0, None).astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000 + extra_ms
        if cache is not None and method == "classical" and not ablate:
            cache["classical"] = (out, aux, ms)
        if return_aux:
            return out, aux, ms
        return out
