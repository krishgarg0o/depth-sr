"""DepthSR - edge-preserving depth upsampling demo (Gradio).

Run locally:   python app.py          -> http://localhost:7860
Docker:        docker build -t depthsr . && docker run -p 7860:7860 depthsr
"""
from __future__ import annotations

import glob
import os
import tempfile
import time
import traceback

import cv2
import gradio as gr
import numpy as np
import plotly.graph_objects as go

from depthsr import classical as C
from depthsr.degrade import make_lr, sparsify
from depthsr.hybrid import METHODS, DepthSR
from depthsr.io_utils import (DEPTH_UNITS, InputError, check_pair, colorize, colorize_map, limit_size,
                              load_depth, to_rgb_uint8)
from depthsr.metrics import all_metrics

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = DepthSR()
MAX_HR_SIDE = 960
NAMES = {"nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic", "jbu": "Joint bilateral",
         "guided": "Guided filter", "classical": "Classical (ours)", "dl": "DL only (TGNet)", "hybrid": "Hybrid (ours)"}
_DA = None   # lazily loaded Depth Anything pipeline (RGB-only mode)


def depth_anything(rgb):
    global _DA
    if _DA is None:
        from transformers import pipeline
        _DA = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    from PIL import Image
    out = _DA(Image.fromarray(rgb))["predicted_depth"]
    rel = out.squeeze().cpu().numpy().astype(np.float32)          # relative inverse depth
    rel = cv2.resize(rel, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-6)
    return (1.0 / (0.1 + rel)).astype(np.float32)                  # pseudo depth, 0.9 .. 10 units


def pointcloud(depth, rgb, title, step):
    H, W = depth.shape
    f = 0.8 * W
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    z = depth[::step, ::step]
    m = z > 0
    X = (xs[m] - W / 2) * z[m] / f
    Y = -(ys[m] - H / 2) * z[m] / f
    col = rgb[::step, ::step][m]
    colors = [f"rgb({r},{g},{b})" for r, g, b in col]
    fig = go.Figure(go.Scatter3d(x=X, y=-z[m], z=Y, mode="markers", marker=dict(size=1.6, color=colors)))
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=30, b=0), height=380,
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                                 camera=dict(eye=dict(x=0.9, y=1.4, z=0.5))))
    return fig


def save_outputs(depth):
    d = tempfile.mkdtemp()
    png = os.path.join(d, "depth_upsampled_mm.png")
    cv2.imwrite(png, np.clip(depth * 1000, 0, 65535).astype(np.uint16))
    npy = os.path.join(d, "depth_upsampled.npy")
    np.save(npy, depth)
    return [png, npy]


def run(rgb_in, depth_file, unit, input_kind, scale, noise, holes, sparse_pts, method, compare_all,
        progress=gr.Progress()):
    t_start = time.perf_counter()
    try:
        if rgb_in is None:
            raise InputError("Please upload an RGB guide image (or pick an example).")
        scale = int(scale)
        rgb = limit_size(to_rgb_uint8(rgb_in), MAX_HR_SIDE)
        gt = None
        note = ""
        progress(0.1, desc="Loading inputs")
        if depth_file is None:
            if input_kind != "RGB only (pseudo-GT demo)":
                raise InputError("No depth file uploaded. Upload a depth map, or choose "
                                 "'RGB only (pseudo-GT demo)' to generate one with Depth Anything V2.")
            progress(0.2, desc="Estimating pseudo ground-truth depth (Depth Anything V2)")
            try:
                gt = depth_anything(rgb)
            except Exception as e:  # network / model download issue
                raise InputError(f"RGB-only mode needs the Depth Anything model, which failed to load: {e}")
            note = ("Pseudo ground truth from Depth Anything V2 (relative units) - "
                    "metrics measure upsampling quality, not metric accuracy.")
        else:
            depth = load_depth(depth_file if isinstance(depth_file, str) else depth_file.name, unit)
            if input_kind == "Low-res depth (upsample it)":
                H, W = rgb.shape[:2]
                h, w = depth.shape
                if h * 2 > H * 1.05 and w * 2 > W * 1.05:
                    raise InputError(f"The depth ({w}x{h}) is not lower resolution than the RGB ({W}x{H}). "
                                     "Choose 'Full-res depth (simulate degradation)' instead.")
                rgb = check_pair(rgb, depth, scale)
                depth_lr = depth
            else:
                if depth.shape != rgb.shape[:2]:
                    ar_r, ar_d = rgb.shape[1] / rgb.shape[0], depth.shape[1] / depth.shape[0]
                    if abs(ar_r - ar_d) / ar_d > 0.05:
                        raise InputError(f"RGB {rgb.shape[1]}x{rgb.shape[0]} and depth "
                                         f"{depth.shape[1]}x{depth.shape[0]} have different aspect ratios.")
                    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                gt = depth
        if gt is not None:
            H, W = gt.shape
            H2, W2 = H // scale * scale, W // scale * scale
            gt, rgb = gt[:H2, :W2], rgb[:H2, :W2]
            if min(H2, W2) // scale < 4:
                raise InputError(f"Image too small for x{scale} upsampling.")
            gt_in = C.fill_holes(gt) if (gt <= 0).any() else gt
            depth_lr = make_lr(gt_in, scale, float(noise), float(holes), seed=0)
            if sparse_pts and int(sparse_pts) > 0:
                depth_lr = sparsify(depth_lr, int(sparse_pts), np.random.default_rng(0))

        if method in ("dl", "hybrid") and not SR.has_weights:
            raise InputError("Trained weights are missing (checkpoints/tgnet.pt). Choose a classical method.")

        progress(0.3, desc="Classical stage")
        cache = {}
        SR.run(depth_lr, rgb, scale, "classical", cache=cache)        # shared by classical / jbu / hybrid
        progress(0.45, desc=f"Running {NAMES[method]}")
        out, aux, ms = SR.run(depth_lr, rgb, scale, method, return_aux=True, cache=cache)
        cls_aux = cache["classical"][1]

        v = (gt if gt is not None else out)
        vmin, vmax = np.percentile(v[v > 0], 1), np.percentile(v[v > 0], 99)
        H, W = rgb.shape[:2]
        lr_vis = colorize(cv2.resize(depth_lr, (W, H), interpolation=cv2.INTER_NEAREST), vmin, vmax)
        out_vis = colorize(out, vmin, vmax)

        # intermediate visualisations
        e_rgb, _ = C.rgb_edges(rgb)
        dmag, _ = C.depth_discontinuities(cls_aux["bicubic"], 0.02)
        stages = [
            (lr_vis, f"1. LR input x{scale} (black = missing)"),
            (colorize_map(cls_aux["hole_mask"]), "2. Missing-depth mask"),
            (colorize(cls_aux["bicubic"], vmin, vmax), "3. Hole-filled + bicubic"),
            (colorize_map(e_rgb), "4. RGB edges"),
            (colorize_map(cls_aux["band"]), "5. Depth-discontinuity band"),
            (colorize_map(cls_aux["trust"]), "6. Trust map (RGB edge AND depth edge)"),
            (colorize_map(cls_aux["texture"]), "7. Texture edges (suppressed)"),
            (colorize(cache["classical"][0], vmin, vmax), "8. Classical refined"),
        ]
        if "residual" in aux:
            r = aux["residual"]
            lim = np.percentile(np.abs(r), 99) + 1e-6
            stages.append((cv2.cvtColor(cv2.applyColorMap(((np.clip(r / lim, -1, 1) + 1) * 127.5).astype(np.uint8),
                                                          cv2.COLORMAP_COOL), cv2.COLOR_BGR2RGB), "9. Network residual"))
            stages.append((colorize_map(aux["alpha"]), "10. Fusion gate alpha (white = trust network)"))
        stages.append((out_vis, f"Final: {NAMES[method]}"))

        # comparison of methods
        rows, gallery = [], []
        methods = [m for m in METHODS if SR.has_weights or m not in ("dl", "hybrid")] if compare_all else [method]
        bic_out = None
        for i, m in enumerate(methods):
            progress(0.5 + 0.4 * i / len(methods), desc=f"Comparing: {NAMES[m]}")
            o, _, t = (out, aux, ms) if m == method else SR.run(depth_lr, rgb, scale, m, return_aux=True, cache=cache)
            if m == "bicubic":
                bic_out = o
            gallery.append((colorize(o, vmin, vmax), f"{NAMES[m]} ({t:.0f} ms)"))
            row = [NAMES[m], round(t, 1)]
            if gt is not None:
                r = all_metrics(o, gt, rgb)
                row += [round(r[k], 4) for k in ("RMSE", "MAE", "PSNR", "SSIM", "BoundaryRMSE", "EdgeF1", "TextureCopy")]
            rows.append(row)
        headers = ["Method", "Latency (ms)"] + (["RMSE", "MAE", "PSNR", "SSIM", "Boundary RMSE", "Edge F1",
                                                 "Texture copy"] if gt is not None else [])

        err_img = None
        if gt is not None:
            err = np.abs(out - gt) * (gt > 0)
            err_img = colorize_map(np.clip(err, 0, np.percentile(err, 99.5)))
        if bic_out is None:
            bic_out = SR.run(depth_lr, rgb, scale, "bicubic")
        step = max(1, int(np.sqrt(H * W / 30000)))
        pc_a = pointcloud(bic_out, rgb, "Bicubic: note 'flying pixels' at edges", step)
        pc_b = pointcloud(out, rgb, f"{NAMES[method]}", step)

        total = (time.perf_counter() - t_start)
        p = cls_aux.get("params", {})
        status = (f"**Done** in {total:.1f} s - {NAMES[method]} took {ms:.0f} ms on {SR.device.upper()}. "
                  f"Output {W}x{H} from {depth_lr.shape[1]}x{depth_lr.shape[0]} (x{scale}). "
                  f"Missing input pixels: {(depth_lr <= 0).mean()*100:.1f} %. "
                  f"Adaptive params: noise={p.get('noise', 0):.4f}, radius={p.get('radius', '-')}, "
                  f"sigma_color={p.get('sigma_color', 0):.1f}. " + note)
        return ((lr_vis, out_vis), stages, gallery, gr.update(value=rows, headers=headers), err_img, pc_a, pc_b,
                save_outputs(out), status)
    except InputError as e:
        raise gr.Error(str(e))
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {type(e).__name__}: {e}")


def examples():
    ex = []
    for rgb in sorted(glob.glob(os.path.join(ROOT, "samples", "*_rgb.png"))):
        dep = rgb.replace("_rgb.png", "_depth.png")
        if os.path.exists(dep):
            ex.append([rgb, dep, "auto", "Full-res depth (simulate degradation)", "8", 0.0, 0.0, 0, "hybrid", True])
    return ex


INTRO = """# DepthSR - Edge-Preserving Depth Upsampling
Turns **low-resolution, sparse or noisy depth** into sharp high-resolution depth using the RGB image as a guide,
without copying RGB texture into the depth. Pipeline: classical hole filling + trust-gated joint bilateral
refinement -> **TGNet** residual network -> learned fusion gate. Pick an example below or upload your own.
"""

HELP = """**Input options**
- *Full-res depth (simulate degradation)*: upload RGB + full-res depth; the app downsamples it (and optionally adds noise / holes / sparsity) and reports metrics against your depth as ground truth.
- *Low-res depth (upsample it)*: upload RGB + a real low-res depth; the depth is upsampled to RGB resolution (no ground truth, so no metrics).
- *RGB only (pseudo-GT demo)*: only an RGB photo; depth is estimated with Depth Anything V2 and used as pseudo ground truth.

Depth formats: 16-bit PNG (mm), KITTI PNG (/256), 8-bit PNG, .npy, .pfm, .tiff.
"""

with gr.Blocks(title="DepthSR - P4 Depth Upsampling") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            rgb_in = gr.Image(label="RGB guide image", type="numpy", height=240)
            depth_in = gr.File(label="Depth map (png / npy / pfm / tiff)", file_types=[".png", ".npy", ".pfm", ".tif", ".tiff"],
                               type="filepath")
            input_kind = gr.Radio(["Full-res depth (simulate degradation)", "Low-res depth (upsample it)",
                                   "RGB only (pseudo-GT demo)"], value="Full-res depth (simulate degradation)",
                                  label="Input type")
            unit = gr.Dropdown(list(DEPTH_UNITS.keys()), value="auto", label="Depth units")
            scale = gr.Radio(["2", "4", "8", "16"], value="8", label="Upsampling factor")
            with gr.Accordion("Degradation (simulate mode)", open=False):
                noise = gr.Slider(0, 0.05, 0.0, step=0.005, label="Depth noise (relative sigma)")
                holes = gr.Slider(0, 0.2, 0.0, step=0.01, label="Missing depth (fraction)")
                sparse = gr.Slider(0, 2000, 0, step=100, label="Sparse: keep N points of LR (0 = off)")
            method = gr.Dropdown([(NAMES[m], m) for m in METHODS], value="hybrid" if SR.has_weights else "classical",
                                 label="Method")
            compare = gr.Checkbox(True, label="Compare all methods")
            btn = gr.Button("Upsample depth", variant="primary")
            ex = examples()
            if ex:
                gr.Examples(ex, [rgb_in, depth_in, unit, input_kind, scale, noise, holes, sparse, method, compare],
                            label="Examples (unseen test images) - click one, then Upsample")
            gr.Markdown(HELP)
        with gr.Column(scale=2):
            status = gr.Markdown("Upload inputs or choose an example, then press **Upsample depth**.")
            slider = gr.ImageSlider(label="Low-res input  |  result (drag)", height=360)
            with gr.Tabs():
                with gr.Tab("Pipeline stages"):
                    stages = gr.Gallery(columns=4, height=420, object_fit="contain")
                with gr.Tab("Method comparison"):
                    comp = gr.Gallery(columns=4, height=360, object_fit="contain")
                    table = gr.Dataframe(label="Metrics (lower RMSE/MAE/Boundary/Texture is better)", wrap=True)
                    err = gr.Image(label="Absolute error of selected method (bright = large)", height=260)
                with gr.Tab("3D point cloud"):
                    with gr.Row():
                        pc_a = gr.Plot()
                        pc_b = gr.Plot()
                with gr.Tab("Download"):
                    files = gr.File(label="Upsampled depth (16-bit PNG in mm, float .npy)", file_count="multiple")
    btn.click(run, [rgb_in, depth_in, unit, input_kind, scale, noise, holes, sparse, method, compare],
              [slider, stages, comp, table, err, pc_a, pc_b, files, status], concurrency_limit=2)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                   theme=gr.themes.Soft())
