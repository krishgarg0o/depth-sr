"""DepthSR - Streamlit front-end (deployed on Streamlit Community Cloud).

Entry point for Streamlit Cloud: streamlit_app/app.py
Local: streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import base64
import glob
import hashlib
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
SRC_EXAMPLE, SRC_UPLOAD, SRC_PHOTO = "Test scene", "My RGB + depth", "Any photo"
KINDS = ["Full-res depth (simulate degradation)", "Low-res depth (upsample it)"]
TURBO = "#30123b,#4662d7,#36aaf9,#1ae4b6,#72fe5e,#c8ef34,#faba39,#f66b19,#ca2a04,#7a0403"

st.set_page_config(page_title="DepthSR - Edge-Preserving Depth Upsampling", page_icon="🧊", layout="wide")

# ----------------------------------------------------------------------------- style
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&display=swap" rel="stylesheet">
<style>
:root {--ink:#241a3b; --paper:#f3f5f7; --line:#d9dee5; --muted:#5b6472; --accent:#e8562a; --good:#13795b; --bad:#b42318;}
.stApp {background: var(--paper);}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1240px;}
h1, h2, h3, h4, .ds-title {font-family: 'Archivo', sans-serif; font-stretch: 85%; color: var(--ink); letter-spacing: -0.01em;}
.ds-top {display:flex; align-items:flex-end; justify-content:space-between; gap:16px; flex-wrap:wrap;
         border-bottom: 2px solid var(--ink); padding-bottom: 10px; margin-bottom: 14px;}
.ds-title {font-size: 2.6rem !important; font-weight: 800; line-height: 1; margin: 0;}
.ds-sub {color: var(--muted); font-size: 1rem; margin: 6px 0 0 0; max-width: 62ch;}
.ds-links a {color: var(--ink); font-weight: 600; text-decoration: none; border-bottom: 2px solid var(--accent);
             margin-left: 14px; font-size: 0.92rem;}
.ds-legend {display:flex; align-items:center; gap:10px; color: var(--muted); font-size: 0.84rem; margin: 8px 0 2px;}
.ds-bar {flex:1; height: 10px; border-radius: 5px; background: linear-gradient(90deg, TURBO_STOPS);}
.ds-read {background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px;}
.ds-read h4 {margin: 0 0 4px 0; font-size: 1.05rem;}
.ds-read .cap {color: var(--muted); font-size: 0.82rem; margin-bottom: 10px;}
.ds-row {display:grid; grid-template-columns: 1fr auto; row-gap: 3px; padding: 8px 0; border-top: 1px solid #eef1f4;}
.ds-row .k {font-size: 0.86rem; color: var(--muted);}
.ds-row .v {font-family: 'Archivo', sans-serif; font-size: 1.35rem; font-weight: 700; color: var(--ink); text-align:right;}
.ds-row .d {grid-column: 1 / 3; display:flex; align-items:center; gap:8px; font-size: 0.78rem; color: var(--muted);}
.ds-track {flex:1; height: 6px; background: #eef1f4; border-radius: 3px; position: relative; overflow: hidden;}
.ds-fill {position:absolute; left:0; top:0; bottom:0; border-radius: 3px;}
.good {color: var(--good); font-weight: 700;} .bad {color: var(--bad); font-weight: 700;}
.ds-flowlabel {font-family:'Archivo',sans-serif; font-weight:700; color: var(--ink); font-size: 1rem; margin: 10px 0 2px;}
.ds-flowcap {color: var(--muted); font-size: 0.84rem; margin: 0 0 8px;}
.ds-note {color: var(--muted); font-size: 0.84rem; line-height: 1.45;}
.ds-foot {margin-top: 26px; padding-top: 10px; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.82rem;}
.ds-foot a {color: var(--ink);}
div[data-testid="stVerticalBlockBorderWrapper"] {background: #fff;}
.stButton button[kind="primary"] {background: var(--accent); border-color: var(--accent);}
@media (max-width: 640px) {.ds-title {font-size: 1.7rem;} .ds-links a {margin: 0 14px 0 0;}}
</style>
""".replace("TURBO_STOPS", TURBO), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- models
@st.cache_resource(show_spinner=False)
def get_model():
    return DepthSR()


@st.cache_resource(show_spinner=False)
def get_depth_estimator():
    """Depth Anything V2 Small (pretrained monocular depth) - only used for 'Any photo'."""
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
def decode_rgb(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("That image could not be read. Use a PNG or JPG file.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_depth(data: bytes, name: str, unit: str):
    suffix = os.path.splitext(name)[1].lower() or ".png"
    path = os.path.join("/tmp", f"upl_{time.time_ns()}{suffix}")
    with open(path, "wb") as fh:
        fh.write(data)
    try:
        return load_depth(path, unit)
    finally:
        os.remove(path)


def to_b64(img):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode()


def compare_slider(left, right, left_label, right_label, height):
    """Drag-to-compare view (pure HTML/CSS, no extra dependency)."""
    h, w = left.shape[:2]
    a, b = to_b64(left), to_b64(right)
    html = f"""
    <div id="wrap" style="position:relative;width:100%;max-width:{int(w * (height - 8) / h)}px;aspect-ratio:{w}/{h};
         margin:0 auto;border-radius:10px;overflow:hidden;user-select:none;font-family:sans-serif;touch-action:none;">
      <img src="data:image/jpeg;base64,{b}" style="position:absolute;inset:0;width:100%;height:100%;">
      <div id="clip" style="position:absolute;inset:0;width:50%;overflow:hidden;">
        <img id="limg" src="data:image/jpeg;base64,{a}" style="position:absolute;top:0;left:0;height:100%;max-width:none;">
      </div>
      <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#fff;">
        <div style="position:absolute;top:50%;left:-15px;width:32px;height:32px;margin-top:-16px;border-radius:50%;
             background:#fff;box-shadow:0 1px 6px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;
             font-size:15px;color:#241a3b;">&#8596;</div></div>
      <span style="position:absolute;left:10px;bottom:10px;background:rgba(36,26,59,.78);color:#fff;padding:3px 9px;
           border-radius:5px;font-size:13px;">{left_label}</span>
      <span style="position:absolute;right:10px;bottom:10px;background:#e8562a;color:#fff;padding:3px 9px;
           border-radius:5px;font-size:13px;">{right_label}</span>
      <input id="rng" type="range" min="0" max="100" value="50" aria-label="Drag to compare"
        style="position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0;">
    </div>
    <script>
      const r=document.getElementById('rng'), c=document.getElementById('clip'), bar=document.getElementById('bar'),
            wrap=document.getElementById('wrap'), img=document.getElementById('limg');
      function upd(){{ c.style.width=r.value+'%'; bar.style.left=r.value+'%'; img.style.width=wrap.clientWidth+'px'; }}
      r.addEventListener('input',upd); window.addEventListener('resize',upd); upd();
    </script>"""
    components.html(html, height=height)


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
    fig.update_layout(title=title, height=380, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)",
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
            raise InputError(f"This depth map ({w}x{h}) is not lower resolution than the photo ({W}x{H}). "
                             "Switch the depth type to 'Full-res depth (simulate degradation)'.")
        rgb = check_pair(rgb, depth, scale)
        depth_lr = depth
    else:
        if depth.shape != rgb.shape[:2]:
            ar_r, ar_d = rgb.shape[1] / rgb.shape[0], depth.shape[1] / depth.shape[0]
            if abs(ar_r - ar_d) / ar_d > 0.05:
                raise InputError(f"The photo ({rgb.shape[1]}x{rgb.shape[0]}) and depth map "
                                 f"({depth.shape[1]}x{depth.shape[0]}) have different shapes. Both must show the same view.")
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        H, W = depth.shape
        H2, W2 = H // scale * scale, W // scale * scale
        if min(H2, W2) // scale < 4:
            raise InputError(f"The image is too small for x{scale}. Pick a smaller factor.")
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


@st.cache_data(max_entries=12, show_spinner=False)
def run_cached(src, key, payload, unit, kind, scale, noise, holes, sparse_pts, method, compare_all):
    """Cached end-to-end run; key identifies the input so tab switches never recompute."""
    if src == SRC_EXAMPLE:
        rgb = cv2.cvtColor(cv2.imread(payload[0]), cv2.COLOR_BGR2RGB)
        depth = load_depth(payload[0].replace("_rgb.png", "_depth.png"), "auto")
    elif src == SRC_UPLOAD:
        rgb = decode_rgb(payload[0])
        depth = decode_depth(payload[1], payload[2], unit)
    else:
        rgb = limit_size(decode_rgb(payload[0]), MAX_HR_SIDE)
        try:
            depth = estimate_depth(rgb)
        except Exception as e:  # noqa: BLE001 - model download / memory problems must not crash the app
            raise InputError("Depth estimation from a photo is unavailable right now "
                             f"({type(e).__name__}). Use a test scene or upload a photo with its depth map.")
        kind = KINDS[0]
    t0 = time.perf_counter()
    R = process(rgb, depth, kind, scale, noise, holes, sparse_pts, method, compare_all)
    R["total"] = time.perf_counter() - t0
    return R


def readout_row(label, ours, base, unit, higher_better, digits):
    """One instrument-style row: value, and a bar comparing ours with bicubic."""
    if base is None:
        return (f'<div class="ds-row"><span class="k">{label}</span>'
                f'<span class="v">{ours:.{digits}f}{unit}</span></div>')
    better = ours > base if higher_better else ours < base
    rel = (ours - base) / (abs(base) + 1e-9) * 100
    mx = max(abs(ours), abs(base), 1e-9)
    col = "#13795b" if better else "#b42318"
    word = "better" if better else "worse"
    return (f'<div class="ds-row"><span class="k">{label}</span><span class="v">{ours:.{digits}f}{unit}</span>'
            f'<div class="d"><span>ours</span><div class="ds-track"><div class="ds-fill" style="width:{abs(ours) / mx * 100:.0f}%;'
            f'background:{col}"></div></div><span class="{"good" if better else "bad"}">{abs(rel):.0f}% {word}</span></div>'
            f'<div class="d"><span>bicubic</span><div class="ds-track"><div class="ds-fill" style="width:{abs(base) / mx * 100:.0f}%;'
            f'background:#9aa3af"></div></div><span>{base:.{digits}f}{unit}</span></div></div>')


# ----------------------------------------------------------------------------- header
st.markdown(f"""
<div class="ds-top">
  <div><div class="ds-title">DepthSR</div>
  <p class="ds-sub">Sharpens low-resolution, noisy or incomplete depth maps. The colour photo is used as a guide only
  where it shows a real depth edge, so object outlines come out sharp and wall patterns stay flat.</p></div>
  <div class="ds-links"><a href="{REPO}" target="_blank">Code</a>
  <a href="{REPO}/blob/main/results/eval_P4_T74.md" target="_blank">Results</a>
  <a href="{REPO}/blob/main/docs/design.md" target="_blank">Design</a></div>
</div>
""", unsafe_allow_html=True)
if not SR.has_weights:
    st.info("Trained network weights are missing, so only the classical methods are available.")

samples = sorted(s for s in glob.glob(os.path.join(ROOT, "samples", "*_rgb.png"))
                 if os.path.exists(s.replace("_rgb.png", "_depth.png")))


def nice_name(path):
    n = os.path.basename(path).replace("_rgb.png", "")
    if n.startswith("nyu"):
        return f"Indoor room {int(n[3:]) + 1} (NYU test set, never seen in training)"
    if n.startswith("synthetic"):
        return f"Synthetic scene {int(n[9:]) + 1} (patterned surfaces)"
    return n


# ----------------------------------------------------------------------------- toolbar
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.7, 1.15, 1.0, 0.85], gap="medium")
    with c1:
        src = st.segmented_control("Input", [SRC_EXAMPLE, SRC_UPLOAD, SRC_PHOTO], default=SRC_EXAMPLE) or SRC_EXAMPLE
    with c2:
        scale = st.segmented_control("Upsampling factor", [2, 4, 8, 16], default=8, format_func=lambda s: f"x{s}") or 8
    with c3:
        avail = [m for m in METHODS if SR.has_weights or m not in ("dl", "hybrid")]
        method = st.selectbox("Method", avail, index=avail.index("hybrid" if SR.has_weights else "classical"),
                              format_func=lambda m: NAMES[m])
    with c4:
        with st.popover("Make it harder", use_container_width=True):
            noise = st.slider("Sensor noise (relative)", 0.0, 0.05, 0.0, 0.005)
            holes = st.slider("Missing depth (fraction of pixels)", 0.0, 0.2, 0.0, 0.01)
            sparse_pts = st.slider("Keep only N depth points (0 = off)", 0, 2000, 0, 100)
            compare_all = st.checkbox("Also run all 8 methods (slower)", False)

    payload, key, unit, kind = None, None, "auto", KINDS[0]
    if src == SRC_EXAMPLE:
        if samples:
            pick = st.selectbox("Scene", samples, format_func=nice_name)
            payload, key = (pick,), pick
        else:
            st.warning("No test scenes are bundled with this deployment.")
    elif src == SRC_UPLOAD:
        u1, u2, u3 = st.columns([1, 1, 1])
        rgb_file = u1.file_uploader("Colour photo", type=["png", "jpg", "jpeg", "bmp"])
        depth_file = u2.file_uploader("Depth map (16-bit PNG in mm, KITTI PNG, 8-bit PNG, .npy, .pfm, .tiff)",
                                      type=["png", "npy", "pfm", "tif", "tiff"])
        kind = u3.radio("Depth type", KINDS)
        unit = u3.selectbox("Depth units", list(DEPTH_UNITS.keys()))
        if rgb_file is not None and depth_file is not None:
            rb, db = rgb_file.getvalue(), depth_file.getvalue()
            payload = (rb, db, depth_file.name)
            key = hashlib.md5(rb + db).hexdigest()
        else:
            st.markdown('<span class="ds-note">Add a colour photo and its depth map to run. For a quick try, '
                        'use a test scene or "Any photo".</span>', unsafe_allow_html=True)
    else:
        photo_file = st.file_uploader("Any photo (JPG or PNG)", type=["png", "jpg", "jpeg", "bmp", "webp"])
        st.markdown('<span class="ds-note">Without a depth sensor, depth is first estimated from the photo with '
                    'Depth Anything V2, then shrunk and rebuilt by our pipeline. Scores compare against that estimate '
                    '(relative units), so they show rebuild quality rather than true distance accuracy. The first photo '
                    'takes longer while the estimator loads.</span>', unsafe_allow_html=True)
        if photo_file is not None:
            pb = photo_file.getvalue()
            payload, key = (pb,), hashlib.md5(pb).hexdigest()

# ----------------------------------------------------------------------------- run
if payload is None:
    st.markdown('<p class="ds-note" style="margin-top:14px">Results appear here as soon as an input is ready.</p>',
                unsafe_allow_html=True)
else:
    try:
        with st.spinner("Estimating depth from the photo, then upsampling..." if src == SRC_PHOTO
                        else f"Upsampling x{scale}..."):
            R = run_cached(src, key, payload, unit, kind, scale, noise, holes, sparse_pts, method, compare_all)
    except InputError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:  # noqa: BLE001
        st.error(f"Processing failed ({type(e).__name__}: {e}). Try another input or a smaller factor.")
        st.stop()

    unit_mul, unit_lbl = (1.0, "") if src == SRC_PHOTO else (100.0, " cm")
    if src == SRC_UPLOAD and unit == "8-bit relative (0-255)":
        unit_lbl = " u"
    rgb, gt, lr, out = R["rgb"], R["gt"], R["depth_lr"], R["out"]
    H, W = rgb.shape[:2]
    v = gt if gt is not None else out
    vmin, vmax = np.percentile(v[v > 0], 1), np.percentile(v[v > 0], 99)
    res = {m: (o, t, r) for m, o, t, r in R["results"]}
    bic_o, _, bic_r = res["bicubic"]
    _, _, sel_r = res[method]

    left, right = st.columns([2.05, 1], gap="large")
    with left:
        compare_slider(colorize(bic_o, vmin, vmax), colorize(out, vmin, vmax),
                       f"Bicubic x{scale}", NAMES[method], height=468)
        lo, hi = vmin * unit_mul, vmax * unit_mul
        st.markdown(f'<div class="ds-legend"><span>near {lo:.0f}{unit_lbl if src != SRC_PHOTO else ""}</span>'
                    f'<div class="ds-bar"></div><span>far {hi:.0f}{unit_lbl if src != SRC_PHOTO else ""}</span></div>'
                    '<p class="ds-note">Drag the handle. Left: plain bicubic interpolation of the low-resolution input. '
                    'Right: our result. Watch the object outlines.</p>', unsafe_allow_html=True)
    with right:
        rows = [f'<div class="ds-read"><h4>{NAMES[method]} at x{scale}</h4>'
                f'<div class="cap">{lr.shape[1]}x{lr.shape[0]} depth in, {W}x{H} out, '
                f'{R["ms"] / 1000:.2f} s on CPU</div>']
        if sel_r is not None:
            nb = method == "bicubic"
            rows.append(readout_row("Depth error (RMSE)", sel_r["RMSE"] * unit_mul,
                                    None if nb else bic_r["RMSE"] * unit_mul, unit_lbl, False, 2))
            rows.append(readout_row("Error at object edges", sel_r["BoundaryRMSE"] * unit_mul,
                                    None if nb else bic_r["BoundaryRMSE"] * unit_mul, unit_lbl, False, 2))
            rows.append(readout_row("Edge sharpness (F1)", sel_r["EdgeF1"], None if nb else bic_r["EdgeF1"], "", True, 3))
            rows.append(readout_row("Structure (SSIM)", sel_r["SSIM"], None if nb else bic_r["SSIM"], "", True, 4))
        else:
            rows.append('<p class="ds-note">This is a real low-resolution input, so there is no ground truth to score '
                        'against. Compare the outlines in the slider and the 3D view.</p>')
        rows.append("</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)
        if src == SRC_PHOTO:
            st.caption("Scores are measured against the depth estimated from the photo.")

    t1, t2, t3, t4, t5 = st.tabs(["How it got there", "All methods", "3D view", "Where the errors are", "Download"])
    with t1:
        ca = R["cls_aux"]
        e_rgb, _ = C.rgb_edges(rgb)
        groups = [
            ("1. What the sensor gives", "A colour photo and a coarse depth map.",
             [(rgb, "Colour photo"),
              (colorize(cv2.resize(lr, (W, H), interpolation=cv2.INTER_NEAREST), vmin, vmax),
               f"Low-res depth, x{scale} smaller (black = missing)"),
              (colorize(ca["bicubic"], vmin, vmax), "Gaps filled, plain bicubic")]),
            ("2. Classical stage", "Photo edges are trusted only where the depth also jumps; pattern edges are ignored.",
             [(colorize_map(e_rgb), "Every edge in the photo"),
              (colorize_map(ca["band"]), "Where the depth jumps"),
              (colorize_map(ca["trust"]), "Trusted edges (both agree)"),
              (colorize_map(ca["texture"]), "Pattern edges, ignored"),
              (colorize(R["cls_out"], vmin, vmax), "Classical result")]),
        ]
        net = []
        if "residual" in R["aux"]:
            r = R["aux"]["residual"]
            lim = np.percentile(np.abs(r), 99) + 1e-6
            net.append((cv2.cvtColor(cv2.applyColorMap(((np.clip(r / lim, -1, 1) + 1) * 127.5).astype(np.uint8),
                                                       cv2.COLORMAP_COOL), cv2.COLOR_BGR2RGB), "Network correction"))
            net.append((colorize_map(R["aux"]["alpha"]), "How much the correction is trusted"))
        net.append((colorize(out, vmin, vmax), f"Final: {NAMES[method]}"))
        if gt is not None:
            net.append((colorize(gt, vmin, vmax), "Reference depth" if src != SRC_PHOTO else "Estimated depth (reference)"))
        groups.append(("3. Network", "A small network corrects the classical result where it is still wrong.", net))
        for title, cap, ims in groups:
            st.markdown(f'<p class="ds-flowlabel">{title}</p><p class="ds-flowcap">{cap}</p>', unsafe_allow_html=True)
            cols = st.columns(min(len(ims), 5))
            for i, (im, c) in enumerate(ims):
                cols[i % len(cols)].image(im, caption=c, use_container_width=True)
    with t2:
        if not compare_all:
            st.markdown('<p class="ds-note">Showing bicubic and the selected method. Turn on "Also run all 8 methods" '
                        'under Make it harder to compare every baseline.</p>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (m, o, t, r) in enumerate(R["results"]):
            cols[i % 4].image(colorize(o, vmin, vmax), caption=f"{NAMES[m]}, {t:.0f} ms", use_container_width=True)
        table = []
        for m, o, t, r in R["results"]:
            row = {"Method": NAMES[m], "Time (ms)": round(t, 1)}
            if r is not None:
                row.update({f"RMSE{unit_lbl}": round(r["RMSE"] * unit_mul, 3), f"MAE{unit_lbl}": round(r["MAE"] * unit_mul, 3),
                            "PSNR": round(r["PSNR"], 2), "SSIM": round(r["SSIM"], 4),
                            f"Edge RMSE{unit_lbl}": round(r["BoundaryRMSE"] * unit_mul, 3),
                            "Edge F1": round(r["EdgeF1"], 4), "Texture copy": round(r["TextureCopy"], 3)})
            table.append(row)
        st.dataframe(table, use_container_width=True, hide_index=True)
    with t3:
        a, b = st.columns(2)
        a.plotly_chart(pointcloud(bic_o, rgb, "Bicubic: points smeared between objects"), use_container_width=True)
        b.plotly_chart(pointcloud(out, rgb, NAMES[method]), use_container_width=True)
    with t4:
        if gt is not None:
            err = np.abs(out - gt) * (gt > 0)
            capv = float(np.percentile(err, 99.5))
            a, b = st.columns(2)
            a.image(colorize_map(np.clip(np.abs(bic_o - gt) * (gt > 0), 0, capv)), caption="Bicubic error",
                    use_container_width=True)
            b.image(colorize_map(np.clip(err, 0, capv)), caption=f"{NAMES[method]} error (same scale, brighter = worse)",
                    use_container_width=True)
        else:
            st.markdown('<p class="ds-note">A real low-resolution input has no reference depth, so there is no error map.</p>',
                        unsafe_allow_html=True)
    with t5:
        ok, png = cv2.imencode(".png", np.clip(out * 1000, 0, 65535).astype(np.uint16))
        d1, d2 = st.columns(2)
        d1.download_button("Download depth (16-bit PNG, mm)", png.tobytes(), "depth_upsampled_mm.png", "image/png",
                           use_container_width=True)
        buf = io.BytesIO()
        np.save(buf, out)
        d2.download_button("Download depth (float .npy)", buf.getvalue(), "depth_upsampled.npy",
                           use_container_width=True)

with st.expander("How DepthSR works"):
    st.markdown("""
- **Classical stage.** Fills holes, measures how noisy the depth is, and sharpens it with a joint bilateral filter guided
  by the photo, but only where a photo edge lines up with a real jump in depth (the trust map). Smoothing is used only
  when the input is actually noisy.
- **Network (TGNet, 2 M parameters).** Looks at the classical result, the photo and the trust map, and predicts a
  correction plus a per-pixel weight for how much to apply it. One model handles x2 to x16.
- **Tested on unseen data.** 100 NYU test rooms: at x8 the error is 4.33 cm against 5.82 cm for bicubic, and edges are
  far sharper (edge F1 0.90 against 0.48). It is also the best method on 26 Middlebury scenes it never trained on.
""")
    arch = os.path.join(ROOT, "docs", "architecture.png")
    if os.path.exists(arch):
        st.image(arch, caption="System architecture", use_container_width=True)

st.markdown(f"""<div class="ds-foot">Team T74: Jidvish Gupta, Keerthana Kiran, Krish Garg, Krishaa Daga.
Computer Vision Hackathon 2026, problem P4. <a href="{REPO}" target="_blank">Source code and results on GitHub</a>.</div>""",
            unsafe_allow_html=True)
