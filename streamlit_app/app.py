"""DepthSR - Streamlit front-end (deployed on Streamlit Community Cloud).

Entry point for Streamlit Cloud: streamlit_app/app.py
Local: streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import base64
import glob
import io
import os
import sys
import time

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depthsr import classical as C  # noqa: E402
from depthsr.degrade import make_lr, sparsify  # noqa: E402
from depthsr.hybrid import METHODS, DepthSR  # noqa: E402
from depthsr.io_utils import (DEPTH_UNITS, InputError, check_pair, colorize, colorize_map, limit_size,  # noqa: E402
                              load_depth)
from depthsr.metrics import all_metrics  # noqa: E402

MAX_HR_SIDE = 960
REPO = "https://github.com/krishgarg0o/depth-sr"
DA_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
NAMES = {"nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic", "jbu": "Joint bilateral",
         "guided": "Guided filter", "classical": "Classical (ours)", "dl": "DL only (TGNet)", "hybrid": "Hybrid (ours)"}
SRC_EXAMPLE, SRC_UPLOAD, SRC_PHOTO = "Example scene", "Upload RGB + depth", "Photo only"
KINDS = ["Full-res depth (simulate degradation)", "Low-res depth (upsample it)"]

st.set_page_config(page_title="DepthSR - Edge-Preserving Depth Upsampling", page_icon="🧊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1300px;}
.hero {background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 55%, #06b6d4 100%); border-radius: 18px;
       padding: 26px 30px 22px; color: white; margin-bottom: 18px;}
.hero h1 {color: white; font-size: 2.05rem; margin: 0 0 6px 0; padding: 0; line-height: 1.2;}
.hero p {color: #e0ecff; font-size: 1.02rem; margin: 0 0 14px 0; max-width: 860px;}
.chips {display: flex; flex-wrap: wrap; gap: 8px;}
.chip {background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
       padding: 5px 12px; font-size: 0.86rem; color: white; white-space: nowrap;}
.chip b {color: #fff;}
.step {font-weight: 600; font-size: 0.95rem; margin: 2px 0 6px 0; color: #1e3a8a;}
.muted {color: #6b7280; font-size: 0.86rem;}
.footer {margin-top: 28px; padding-top: 12px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.85rem;}
.footer a {color: #2563eb; text-decoration: none;}
div[data-testid="stMetricValue"] {font-size: 1.55rem;}
@media (max-width: 640px) {.hero {padding: 18px 18px 16px;} .hero h1 {font-size: 1.45rem;} .chip {font-size: 0.78rem;}}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model():
    return DepthSR()


@st.cache_resource(show_spinner=False)
def get_depth_estimator():
    """Depth Anything V2 Small (pretrained monocular depth) - only used in 'Photo only' mode."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(DA_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DA_MODEL).eval()
    return proc, model


def estimate_depth(rgb):
    """Pseudo ground-truth depth from a single photo (relative units, ~1-10)."""
    import torch
    from PIL import Image
    proc, model = get_depth_estimator()
    inputs = proc(images=Image.fromarray(rgb), return_tensors="pt")
    with torch.no_grad():
        pred = model(**inputs).predicted_depth
    rel = pred.squeeze().cpu().numpy().astype(np.float32)          # relative inverse depth
    rel = cv2.resize(rel, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-6)
    return (1.0 / (0.1 + rel)).astype(np.float32)


SR = get_model()


# ----------------------------------------------------------------------------- helpers
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
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("Could not decode the RGB image (use PNG or JPG).")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_b64(img):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def compare_slider(left, right, left_label, right_label):
    """Drag-to-compare view (pure HTML/CSS, no extra dependency)."""
    h, w = left.shape[:2]
    a, b = to_b64(left), to_b64(right)
    html = f"""
    <div id="wrap" style="position:relative;width:100%;max-width:{w}px;aspect-ratio:{w}/{h};margin:auto;
         border-radius:12px;overflow:hidden;user-select:none;font-family:sans-serif;">
      <img src="data:image/jpeg;base64,{b}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;">
      <div id="clip" style="position:absolute;inset:0;width:50%;overflow:hidden;">
        <img src="data:image/jpeg;base64,{a}" style="position:absolute;top:0;left:0;height:100%;width:auto;max-width:none;">
      </div>
      <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:3px;background:white;
           box-shadow:0 0 6px rgba(0,0,0,.6);"></div>
      <span style="position:absolute;left:10px;top:10px;background:rgba(0,0,0,.6);color:#fff;padding:3px 9px;
           border-radius:6px;font-size:13px;">{left_label}</span>
      <span style="position:absolute;right:10px;top:10px;background:rgba(30,58,138,.85);color:#fff;padding:3px 9px;
           border-radius:6px;font-size:13px;">{right_label}</span>
      <input id="rng" type="range" min="0" max="100" value="50"
        style="position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0;">
    </div>
    <script>
      const r=document.getElementById('rng'), c=document.getElementById('clip'), bar=document.getElementById('bar'),
            wrap=document.getElementById('wrap'), img=c.querySelector('img');
      function upd(){{ c.style.width=r.value+'%'; bar.style.left=r.value+'%'; img.style.width=wrap.clientWidth+'px'; }}
      r.addEventListener('input',upd); window.addEventListener('resize',upd); upd();
    </script>"""
    components.html(html, height=int(min(w, 760) * h / w) + 12)


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

    wanted = [m for m in METHODS if SR.has_weights or m not in ("dl", "hybrid")] if compare_all else \
        list(dict.fromkeys(["bicubic", method]))
    results = []
    for m in wanted:
        o, _, t = (out, aux, ms) if m == method else SR.run(depth_lr, rgb, scale, m, return_aux=True, cache=cache)
        results.append((m, o, t, all_metrics(o, gt, rgb) if gt is not None else None))
    return dict(rgb=rgb, gt=gt, depth_lr=depth_lr, out=out, aux=aux, ms=ms, cls_out=cls_out, cls_aux=cls_aux,
                results=results)


# ----------------------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>🧊 DepthSR &mdash; Edge-Preserving Depth Upsampling</h1>
  <p>Turns low-resolution, sparse or noisy depth into sharp high-resolution depth, using the RGB image as a guide
     <b>only where it marks a real depth edge</b>, so object boundaries stay sharp and RGB texture is not copied.</p>
  <div class="chips">
    <span class="chip">x8 RMSE <b>4.33 cm</b> vs 5.82 bicubic</span>
    <span class="chip">Edge F1 <b>0.90</b> vs 0.48 bicubic</span>
    <span class="chip">Best on unseen <b>Middlebury</b></span>
    <span class="chip">One model for <b>x2&ndash;x16</b></span>
    <span class="chip">Team <b>T74</b> &middot; P4</span>
  </div>
</div>
""", unsafe_allow_html=True)
if not SR.has_weights:
    st.info("Trained TGNet weights are not deployed - classical methods only.")

samples = sorted(s for s in glob.glob(os.path.join(ROOT, "samples", "*_rgb.png"))
                 if os.path.exists(s.replace("_rgb.png", "_depth.png")))


def nice_name(path):
    n = os.path.basename(path).replace("_rgb.png", "")
    if n.startswith("nyu"):
        return f"NYU indoor test scene {int(n[3:]) + 1} (unseen)"
    if n.startswith("synthetic"):
        return f"Synthetic scene {int(n[9:]) + 1} (texture-copy test)"
    return n


left, right = st.columns([1, 1.9], gap="large")

# ----------------------------------------------------------------------------- controls
with left:
    with st.container(border=True):
        st.markdown('<div class="step">1 &middot; Choose input</div>', unsafe_allow_html=True)
        src = st.radio("Input", [SRC_EXAMPLE, SRC_UPLOAD, SRC_PHOTO], horizontal=True, label_visibility="collapsed")
        rgb_file = depth_file = photo_file = pick = None
        kind, unit = KINDS[0], "auto"
        if src == SRC_EXAMPLE:
            if samples:
                pick = st.selectbox("Example", samples, format_func=nice_name)
                st.image(cv2.cvtColor(cv2.imread(pick), cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.warning("No example images bundled.")
        elif src == SRC_UPLOAD:
            rgb_file = st.file_uploader("RGB guide image", type=["png", "jpg", "jpeg", "bmp"])
            depth_file = st.file_uploader("Depth map (16-bit PNG mm, KITTI PNG, 8-bit PNG, .npy, .pfm, .tiff)",
                                          type=["png", "npy", "pfm", "tif", "tiff"])
            kind = st.radio("Depth input type", KINDS)
            unit = st.selectbox("Depth units", list(DEPTH_UNITS.keys()))
        else:
            photo_file = st.file_uploader("Any photo (JPG / PNG)", type=["png", "jpg", "jpeg", "bmp", "webp"])
            st.markdown('<span class="muted">No depth sensor? Depth is first <b>estimated</b> from the photo with '
                        'Depth Anything V2 (pretrained), then downsampled and reconstructed by our pipeline. Metrics '
                        'measure reconstruction quality against that estimate (relative units), not metric '
                        'accuracy. The first run downloads the model (~100 MB) and takes longer.</span>',
                        unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="step">2 &middot; Settings</div>', unsafe_allow_html=True)
        scale = int(st.select_slider("Upsampling factor", options=[2, 4, 8, 16], value=8))
        avail = [m for m in METHODS if SR.has_weights or m not in ("dl", "hybrid")]
        method = st.selectbox("Method", avail, index=avail.index("hybrid" if SR.has_weights else "classical"),
                              format_func=lambda m: NAMES[m])
        with st.expander("Make it harder: noise, holes, sparsity"):
            noise = st.slider("Depth noise (relative sigma)", 0.0, 0.05, 0.0, 0.005)
            holes = st.slider("Missing depth (fraction)", 0.0, 0.2, 0.0, 0.01)
            sparse_pts = st.slider("Sparse: keep N points of the LR depth (0 = off)", 0, 2000, 0, 100)
        compare_all = st.checkbox("Compare all 8 methods (slower)", False)
    go_btn = st.button("▶  Upsample depth", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------- results
with right:
    if not go_btn:
        st.markdown("#### How it works")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**1. Classical stage**  \nHole filling, noise estimate, RGB-guided joint bilateral filter "
                    "applied only where the **trust map** says an RGB edge is a real depth edge.")
        c2.markdown("**2. TGNet**  \nA 2 M-parameter guided U-Net predicts a residual correction and a per-pixel "
                    "**fusion gate** α that decides how much to trust it.")
        c3.markdown("**3. Result**  \nSharp boundaries, no texture copying, robust to noise and holes - from one "
                    "model for x2 to x16.")
        arch = os.path.join(ROOT, "docs", "architecture.png")
        if os.path.exists(arch):
            with st.expander("Architecture diagram"):
                st.image(arch, use_container_width=True)
        st.info("⬅️ Pick an example (or upload your own) and press **Upsample depth**. "
                "Try x16 or add noise and holes to see where the hybrid shines.")
    else:
        try:
            with st.status("Running the pipeline...", expanded=False) as status:
                unit_mul, unit_lbl = 100.0, "cm"
                if src == SRC_EXAMPLE:
                    if pick is None:
                        raise InputError("No example available - upload your own images.")
                    rgb = cv2.cvtColor(cv2.imread(pick), cv2.COLOR_BGR2RGB)
                    depth = load_depth(pick.replace("_rgb.png", "_depth.png"), "auto")
                elif src == SRC_UPLOAD:
                    if rgb_file is None or depth_file is None:
                        raise InputError("Please upload both an RGB image and a depth map, or use 'Photo only'.")
                    rgb = read_upload_rgb(rgb_file)
                    depth = read_upload_depth(depth_file, unit)
                    unit_mul, unit_lbl = 100.0, "cm" if unit != "8-bit relative (0-255)" else "units"
                else:
                    if photo_file is None:
                        raise InputError("Please upload a photo.")
                    rgb = limit_size(read_upload_rgb(photo_file), MAX_HR_SIDE)
                    status.update(label="Estimating depth from the photo (Depth Anything V2)...")
                    try:
                        depth = estimate_depth(rgb)
                    except Exception as e:  # noqa: BLE001 - model download / memory problems must not crash the app
                        raise InputError("Depth estimation is unavailable right now "
                                         f"({type(e).__name__}). Please use an example or upload RGB + depth.")
                    kind = KINDS[0]
                    unit_mul, unit_lbl = 1.0, "rel. units"
                status.update(label=f"Running {NAMES[method]} at x{scale}...")
                t0 = time.perf_counter()
                R = process(rgb, depth, kind, scale, noise, holes, sparse_pts, method, compare_all)
                total = time.perf_counter() - t0
                status.update(label=f"Done in {total:.1f} s", state="complete")
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
        res = {m: (o, t, r) for m, o, t, r in R["results"]}
        bic_o, _, bic_r = res["bicubic"]
        _, _, sel_r = res[method]

        st.success(f"**{NAMES[method]}** reconstructed {W}x{H} depth from {lr.shape[1]}x{lr.shape[0]} "
                   f"(x{scale}) in **{R['ms'] / 1000:.2f} s**.")
        if sel_r is not None:
            k1, k2, k3, k4 = st.columns(4)
            d_rmse = (sel_r["RMSE"] - bic_r["RMSE"]) * unit_mul
            k1.metric(f"RMSE ({unit_lbl})", f"{sel_r['RMSE'] * unit_mul:.2f}",
                      f"{d_rmse:+.2f} vs bicubic" if method != "bicubic" else None, delta_color="inverse")
            k2.metric(f"Boundary RMSE ({unit_lbl})", f"{sel_r['BoundaryRMSE'] * unit_mul:.2f}",
                      f"{(sel_r['BoundaryRMSE'] - bic_r['BoundaryRMSE']) * unit_mul:+.2f}" if method != "bicubic"
                      else None, delta_color="inverse")
            k3.metric("Edge F1", f"{sel_r['EdgeF1']:.3f}",
                      f"{sel_r['EdgeF1'] - bic_r['EdgeF1']:+.3f}" if method != "bicubic" else None)
            k4.metric("SSIM", f"{sel_r['SSIM']:.4f}",
                      f"{sel_r['SSIM'] - bic_r['SSIM']:+.4f}" if method != "bicubic" else None)

        st.markdown("##### Drag to compare &nbsp;<span class='muted'>bicubic (left) vs "
                    f"{NAMES[method]} (right)</span>", unsafe_allow_html=True)
        compare_slider(colorize(bic_o, vmin, vmax), colorize(out, vmin, vmax), "Bicubic", NAMES[method])

        t1, t2, t3, t4, t5 = st.tabs(["🧩 Pipeline stages", "📊 Methods", "🌐 3D view", "🔥 Error map", "⬇️ Download"])
        with t1:
            ca = R["cls_aux"]
            e_rgb, _ = C.rgb_edges(rgb)
            stages = [
                (rgb, "RGB guide"),
                (colorize(cv2.resize(lr, (W, H), interpolation=cv2.INTER_NEAREST), vmin, vmax),
                 f"Low-res input x{scale} (black = missing)"),
                (colorize(ca["bicubic"], vmin, vmax), "Hole-filled + bicubic"),
                (colorize_map(e_rgb), "RGB edges (all of them)"),
                (colorize_map(ca["band"]), "Depth-discontinuity band"),
                (colorize_map(ca["trust"]), "Trust map = RGB edge × band"),
                (colorize_map(ca["texture"]), "Texture edges (ignored)"),
                (colorize(R["cls_out"], vmin, vmax), "Classical stage output"),
            ]
            if "residual" in R["aux"]:
                r = R["aux"]["residual"]
                lim = np.percentile(np.abs(r), 99) + 1e-6
                stages.append((cv2.cvtColor(cv2.applyColorMap(((np.clip(r / lim, -1, 1) + 1) * 127.5)
                                                              .astype(np.uint8), cv2.COLORMAP_COOL), cv2.COLOR_BGR2RGB),
                               "Network residual r"))
                stages.append((colorize_map(R["aux"]["alpha"]), "Fusion gate α (bright = trust network)"))
            stages.append((colorize(out, vmin, vmax), f"Final: {NAMES[method]}"))
            if gt is not None:
                stages.append((colorize(gt, vmin, vmax), "Ground truth" if src != SRC_PHOTO else "Estimated depth (pseudo-GT)"))
            cols = st.columns(3)
            for i, (im, cap) in enumerate(stages):
                cols[i % 3].image(im, caption=f"{i + 1}. {cap}", use_container_width=True)
        with t2:
            cols = st.columns(4)
            for i, (m, o, t, r) in enumerate(R["results"]):
                cols[i % 4].image(colorize(o, vmin, vmax), caption=f"{NAMES[m]} · {t:.0f} ms", use_container_width=True)
            rows = []
            for m, o, t, r in R["results"]:
                row = {"Method": NAMES[m], "Latency (ms)": round(t, 1)}
                if r is not None:
                    row.update({f"RMSE ({unit_lbl})": round(r["RMSE"] * unit_mul, 3),
                                f"MAE ({unit_lbl})": round(r["MAE"] * unit_mul, 3),
                                "PSNR": round(r["PSNR"], 2), "SSIM": round(r["SSIM"], 4),
                                f"Boundary RMSE ({unit_lbl})": round(r["BoundaryRMSE"] * unit_mul, 3),
                                "Edge F1": round(r["EdgeF1"], 4), "Texture copy": round(r["TextureCopy"], 3)})
                rows.append(row)
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if not compare_all:
                st.caption("Tick 'Compare all 8 methods' in Settings to add every baseline.")
            else:
                st.caption("Lower RMSE / MAE / boundary RMSE / texture copy is better; higher PSNR / SSIM / edge F1 is better.")
        with t3:
            x, y = st.columns(2)
            x.plotly_chart(pointcloud(bic_o, rgb, "Bicubic: 'flying pixels' at edges"), use_container_width=True)
            y.plotly_chart(pointcloud(out, rgb, NAMES[method]), use_container_width=True)
        with t4:
            if gt is not None:
                err = np.abs(out - gt) * (gt > 0)
                cap = float(np.percentile(err, 99.5))
                a, b = st.columns(2)
                a.image(colorize_map(np.clip(np.abs(bic_o - gt) * (gt > 0), 0, cap)), caption="Bicubic |error|",
                        use_container_width=True)
                b.image(colorize_map(np.clip(err, 0, cap)), caption=f"{NAMES[method]} |error| (same scale)",
                        use_container_width=True)
            else:
                st.info("No ground truth for a real low-resolution input, so there is no error map.")
        with t5:
            ok, png = cv2.imencode(".png", np.clip(out * 1000, 0, 65535).astype(np.uint16))
            st.download_button("Depth as 16-bit PNG (millimetres)", png.tobytes(), "depth_upsampled_mm.png", "image/png")
            buf = io.BytesIO()
            np.save(buf, out)
            st.download_button("Depth as float32 .npy (input units)", buf.getvalue(), "depth_upsampled.npy")

st.markdown(f"""<div class="footer">DepthSR · Team T74 (Jidvish Gupta, Keerthana Kiran, Krish Garg, Krishaa Daga) ·
Computer Vision Hackathon 2026, P4 · <a href="{REPO}" target="_blank">GitHub repository</a> ·
<a href="{REPO}/blob/main/results/eval_P4_T74.md" target="_blank">evaluation results</a></div>""",
            unsafe_allow_html=True)
