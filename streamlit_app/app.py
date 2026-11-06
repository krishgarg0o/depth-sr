"""DepthSR - Streamlit front-end (deployed on Streamlit Community Cloud).

Entry point for Streamlit Cloud: streamlit_app/app.py
Local: streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import glob
import io
import os
import sys
import time

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depthsr import classical as C  # noqa: E402
from depthsr.degrade import make_lr, sparsify  # noqa: E402
from depthsr.hybrid import METHODS, DepthSR  # noqa: E402
from depthsr.io_utils import (DEPTH_UNITS, InputError, check_pair, colorize, colorize_map, limit_size,  # noqa: E402
                              load_depth)
from depthsr.metrics import all_metrics  # noqa: E402

MAX_HR_SIDE = 960
NAMES = {"nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic", "jbu": "Joint bilateral",
         "guided": "Guided filter", "classical": "Classical (ours)", "dl": "DL only (TGNet)", "hybrid": "Hybrid (ours)"}
KINDS = ["Full-res depth (simulate degradation)", "Low-res depth (upsample it)"]

st.set_page_config(page_title="DepthSR - Depth Upsampling", page_icon="🧊", layout="wide")


@st.cache_resource
def get_model():
    return DepthSR()


SR = get_model()


def read_upload_depth(f, unit):
    suffix = os.path.splitext(f.name)[1].lower() or ".png"
    path = os.path.join("/tmp", f"upl_{time.time_ns()}{suffix}")
    with open(path, "wb") as fh:
        fh.write(f.getvalue())
    try:
        return load_depth(path, unit)
    finally:
        os.remove(path)


def read_upload_rgb(f):
    data = np.frombuffer(f.getvalue(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("Could not decode the RGB image (use PNG or JPG).")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pointcloud(depth, rgb, title):
    H, W = depth.shape
    step = max(1, int(np.sqrt(H * W / 25000)))
    f = 0.8 * W
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    z = depth[::step, ::step]
    m = z > 0
    X = (xs[m] - W / 2) * z[m] / f
    Y = -(ys[m] - H / 2) * z[m] / f
    col = rgb[::step, ::step][m]
    fig = go.Figure(go.Scatter3d(x=X, y=-z[m], z=Y, mode="markers",
                                 marker=dict(size=1.5, color=[f"rgb({r},{g},{b})" for r, g, b in col])))
    fig.update_layout(title=title, height=380, margin=dict(l=0, r=0, t=30, b=0),
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                                 camera=dict(eye=dict(x=0.9, y=1.4, z=0.5))))
    return fig


def process(rgb, depth, kind, scale, noise, holes, sparse_pts, method, compare_all):
    rgb = limit_size(rgb, MAX_HR_SIDE)
    gt = None
    if kind == KINDS[1]:
        H, W = rgb.shape[:2]
        h, w = depth.shape
        if h * 2 > H * 1.05 and w * 2 > W * 1.05:
            raise InputError(f"The depth ({w}x{h}) is not lower resolution than the RGB ({W}x{H}). "
                             "Choose 'Full-res depth (simulate degradation)'.")
        rgb = check_pair(rgb, depth, scale)
        depth_lr = depth
    else:
        if depth.shape != rgb.shape[:2]:
            ar_r, ar_d = rgb.shape[1] / rgb.shape[0], depth.shape[1] / depth.shape[0]
            if abs(ar_r - ar_d) / ar_d > 0.05:
                raise InputError(f"RGB {rgb.shape[1]}x{rgb.shape[0]} and depth {depth.shape[1]}x{depth.shape[0]} "
                                 "have different aspect ratios - they must show the same view.")
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        H, W = depth.shape
        H2, W2 = H // scale * scale, W // scale * scale
        if min(H2, W2) // scale < 4:
            raise InputError(f"Image too small for x{scale} upsampling.")
        gt, rgb = depth[:H2, :W2], rgb[:H2, :W2]
        gt_in = C.fill_holes(gt) if (gt <= 0).any() else gt
        depth_lr = make_lr(gt_in, scale, noise, holes, seed=0)
        if sparse_pts > 0:
            depth_lr = sparsify(depth_lr, sparse_pts, np.random.default_rng(0))

    cache = {}
    SR.run(depth_lr, rgb, scale, "classical", cache=cache)
    out, aux, ms = SR.run(depth_lr, rgb, scale, method, return_aux=True, cache=cache)
    cls_out, cls_aux, _ = cache["classical"]

    methods = [m for m in METHODS if SR.has_weights or m not in ("dl", "hybrid")] if compare_all else [method]
    results = []
    for m in methods:
        o, _, t = (out, aux, ms) if m == method else SR.run(depth_lr, rgb, scale, m, return_aux=True, cache=cache)
        row = {"Method": NAMES[m], "Latency (ms)": round(t, 1)}
        if gt is not None:
            r = all_metrics(o, gt, rgb)
            row.update({k: round(v, 4) for k, v in r.items()})
        results.append((m, o, row))
    return dict(rgb=rgb, gt=gt, depth_lr=depth_lr, out=out, aux=aux, ms=ms, cls_out=cls_out, cls_aux=cls_aux,
                results=results)


# ----------------------------------------------------------------------------- UI
st.title("🧊 DepthSR - Edge-Preserving Depth Upsampling")
st.markdown(
    "Turns **low-resolution, sparse or noisy depth** into sharp high-resolution depth using the RGB image as a "
    "guide - without copying RGB texture into the depth. Pipeline: classical hole filling + trust-gated joint "
    "bilateral refinement → **TGNet** residual network → learned fusion gate.")
if not SR.has_weights:
    st.info("Trained TGNet weights are not deployed yet - classical methods are available; "
            "'DL only' and 'Hybrid' appear once checkpoints/tgnet.pt is added.")

samples = sorted(glob.glob(os.path.join(ROOT, "samples", "*_rgb.png")))
samples = [s for s in samples if os.path.exists(s.replace("_rgb.png", "_depth.png"))]
c1, c2 = st.columns([1, 2], gap="large")
with c1:
    src = st.radio("Input", ["Example (unseen test image)", "Upload my own"], horizontal=True)
    if src.startswith("Example"):
        if not samples:
            st.warning("No example images bundled.")
        names = [os.path.basename(s).replace("_rgb.png", "") for s in samples]
        pick = st.selectbox("Example", names) if names else None
        rgb_file = depth_file = None
        kind = KINDS[0]
        unit = "auto"
    else:
        rgb_file = st.file_uploader("RGB guide image", type=["png", "jpg", "jpeg", "bmp"])
        depth_file = st.file_uploader("Depth map (16-bit PNG mm, KITTI PNG, 8-bit PNG, .npy, .pfm, .tiff)",
                                      type=["png", "npy", "pfm", "tif", "tiff"])
        kind = st.radio("Depth input type", KINDS)
        unit = st.selectbox("Depth units", list(DEPTH_UNITS.keys()))
    scale = int(st.select_slider("Upsampling factor", options=[2, 4, 8, 16], value=8))
    with st.expander("Degradation (simulate mode)"):
        noise = st.slider("Depth noise (relative sigma)", 0.0, 0.05, 0.0, 0.005)
        holes = st.slider("Missing depth (fraction)", 0.0, 0.2, 0.0, 0.01)
        sparse_pts = st.slider("Sparse: keep N points of the LR depth (0 = off)", 0, 2000, 0, 100)
    avail = [m for m in METHODS if SR.has_weights or m not in ("dl", "hybrid")]
    method = st.selectbox("Method", avail, index=avail.index("hybrid" if SR.has_weights else "classical"),
                          format_func=lambda m: NAMES[m])
    compare_all = st.checkbox("Compare all methods", True)
    go_btn = st.button("Upsample depth", type="primary", use_container_width=True)

with c2:
    if not go_btn:
        st.markdown("⬅️ Pick an example or upload an RGB image + depth map, then press **Upsample depth**.")
        with st.expander("Input options explained", expanded=True):
            st.markdown(
                "- **Full-res depth (simulate degradation)**: the depth is downsampled by the chosen factor "
                "(plus optional noise / holes / sparsity) and reconstructed; your depth is the ground truth, so "
                "metrics are reported.\n"
                "- **Low-res depth (upsample it)**: a real low-res depth is upsampled to the RGB resolution "
                "(no ground truth, so no metrics).")
        st.stop()
    try:
        with st.spinner("Loading inputs..."):
            if src.startswith("Example"):
                if pick is None:
                    raise InputError("No example available - upload your own images.")
                p = os.path.join(ROOT, "samples", pick)
                rgb = cv2.cvtColor(cv2.imread(p + "_rgb.png"), cv2.COLOR_BGR2RGB)
                depth = load_depth(p + "_depth.png", "auto")
            else:
                if rgb_file is None or depth_file is None:
                    raise InputError("Please upload both an RGB image and a depth map.")
                rgb = read_upload_rgb(rgb_file)
                depth = read_upload_depth(depth_file, unit)
        t0 = time.perf_counter()
        with st.spinner(f"Running {NAMES[method]}{' and comparing all methods' if compare_all else ''}..."):
            R = process(rgb, depth, kind, scale, noise, holes, sparse_pts, method, compare_all)
        total = time.perf_counter() - t0
    except InputError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Processing failed: {type(e).__name__}: {e}")
        st.stop()

    rgb, gt, lr, out = R["rgb"], R["gt"], R["depth_lr"], R["out"]
    H, W = rgb.shape[:2]
    v = gt if gt is not None else out
    vmin, vmax = np.percentile(v[v > 0], 1), np.percentile(v[v > 0], 99)
    p = R["cls_aux"]["params"]
    st.success(f"Done in {total:.1f} s - {NAMES[method]} took {R['ms']:.0f} ms. Output {W}x{H} from "
               f"{lr.shape[1]}x{lr.shape[0]} (x{scale}); missing input pixels {(lr <= 0).mean() * 100:.1f} %. "
               f"Adaptive params: noise={p['noise']:.4f}, radius={p['radius']}, sigma_color={p['sigma_color']:.1f}.")
    a, b, c = st.columns(3)
    a.image(rgb, caption="RGB guide", use_container_width=True)
    b.image(colorize(cv2.resize(lr, (W, H), interpolation=cv2.INTER_NEAREST), vmin, vmax),
            caption=f"Low-res input x{scale} (black = missing)", use_container_width=True)
    c.image(colorize(out, vmin, vmax), caption=f"Result: {NAMES[method]}", use_container_width=True)

    t1, t2, t3, t4 = st.tabs(["Pipeline stages", "Method comparison", "3D point cloud", "Download"])
    with t1:
        ca = R["cls_aux"]
        e_rgb, _ = C.rgb_edges(rgb)
        stages = [
            (colorize_map(ca["hole_mask"]), "Missing-depth mask"),
            (colorize(ca["bicubic"], vmin, vmax), "Hole-filled + bicubic"),
            (colorize_map(e_rgb), "RGB edges"),
            (colorize_map(ca["band"]), "Depth-discontinuity band"),
            (colorize_map(ca["trust"]), "Trust map (RGB edge AND depth edge)"),
            (colorize_map(ca["texture"]), "Texture edges (suppressed)"),
            (colorize(R["cls_out"], vmin, vmax), "Classical refined"),
        ]
        if "residual" in R["aux"]:
            r = R["aux"]["residual"]
            lim = np.percentile(np.abs(r), 99) + 1e-6
            stages.append((cv2.cvtColor(cv2.applyColorMap(((np.clip(r / lim, -1, 1) + 1) * 127.5).astype(np.uint8),
                                                          cv2.COLORMAP_COOL), cv2.COLOR_BGR2RGB), "Network residual"))
            stages.append((colorize_map(R["aux"]["alpha"]), "Fusion gate alpha (bright = trust network)"))
        cols = st.columns(3)
        for i, (im, cap) in enumerate(stages):
            cols[i % 3].image(im, caption=f"{i + 1}. {cap}", use_container_width=True)
    with t2:
        cols = st.columns(4)
        for i, (m, o, row) in enumerate(R["results"]):
            cols[i % 4].image(colorize(o, vmin, vmax), caption=f"{NAMES[m]} ({row['Latency (ms)']:.0f} ms)",
                              use_container_width=True)
        st.dataframe([row for _, _, row in R["results"]], use_container_width=True, hide_index=True)
        if gt is not None:
            st.caption("Lower RMSE / MAE / BoundaryRMSE / TextureCopy is better; higher PSNR / SSIM / EdgeF1 is better.")
            err = np.abs(out - gt) * (gt > 0)
            st.image(colorize_map(np.clip(err, 0, np.percentile(err, 99.5))),
                     caption=f"Absolute error of {NAMES[method]} (bright = large)", use_container_width=True)
    with t3:
        bic = next((o for m, o, _ in R["results"] if m == "bicubic"), None)
        if bic is None:
            bic = SR.run(lr, rgb, scale, "bicubic")
        x, y = st.columns(2)
        x.plotly_chart(pointcloud(bic, rgb, "Bicubic: 'flying pixels' at edges"), use_container_width=True)
        y.plotly_chart(pointcloud(out, rgb, NAMES[method]), use_container_width=True)
    with t4:
        ok, png = cv2.imencode(".png", np.clip(out * 1000, 0, 65535).astype(np.uint16))
        st.download_button("Depth as 16-bit PNG (millimetres)", png.tobytes(), "depth_upsampled_mm.png", "image/png")
        buf = io.BytesIO()
        np.save(buf, out)
        st.download_button("Depth as float32 .npy (input units)", buf.getvalue(), "depth_upsampled.npy")
