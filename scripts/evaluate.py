"""Full comparative evaluation, ablation and failure-case analysis.

  python scripts/evaluate.py --dataset nyu --data data --team T01 --usns "1XX...,1XX..."
  python scripts/evaluate.py --dataset middlebury --data middlebury     # unseen generalisation
  python scripts/evaluate.py --dataset synthetic --limit 8              # CPU smoke test

Outputs (results/):
  metrics_<dataset>.json        every number below
  eval_P4_<team>.md             submission file (first line = USNs)
  ablation_<dataset>.md         contribution of each pipeline component
  failure_cases_<dataset>.md    worst images with likely causes + figures in results/figs/
  figs/compare_*.png            visual comparisons with boundary zoom-ins
"""
import argparse
import glob
import json
import os
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depthsr import classical as C  # noqa: E402
from depthsr.degrade import make_lr  # noqa: E402
from depthsr.hybrid import METHODS, DepthSR  # noqa: E402
from depthsr.io_utils import colorize, load_depth  # noqa: E402
from depthsr.metrics import all_metrics  # noqa: E402
from depthsr.synthetic import make_scene  # noqa: E402

METRIC_KEYS = ["RMSE", "MAE", "PSNR", "SSIM", "BoundaryRMSE", "EdgeF1", "TextureCopy"]
NAMES = {"nearest": "Nearest", "bilinear": "Bilinear", "bicubic": "Bicubic", "jbu": "Joint Bilateral (JBU)",
         "guided": "Guided Filter", "classical": "Classical pipeline (ours)", "dl": "Deep learning only (TGNet)",
         "hybrid": "Hybrid (ours)"}
GROUP = {"nearest": "Interpolation", "bilinear": "Interpolation", "bicubic": "Interpolation",
         "jbu": "Classical refinement", "guided": "Classical refinement", "classical": "Classical refinement",
         "dl": "Deep learning", "hybrid": "Hybrid"}


# ----------------------------------------------------------------------------- datasets
def crop16(rgb, d):
    H, W = d.shape
    h, w = H // 16 * 16, W // 16 * 16
    y, x = (H - h) // 2, (W - w) // 2
    return rgb[y:y + h, x:x + w], d[y:y + h, x:x + w]


def load_dataset(name, root, limit):
    if name == "nyu":
        z = np.load(os.path.join(root, "nyu_test.npz"))
        rgb, dep = z["rgb"], z["depth"]
        n = len(rgb) if limit <= 0 else min(limit, len(rgb))
        return [(f"nyu_{1000+i}", rgb[i], dep[i]) for i in range(n)], 100.0, "cm"
    if name == "synthetic":
        n = limit if limit > 0 else 16
        return [(f"syn_{i}", *crop16(*make_scene(240, 320, seed=1000 + i))) for i in range(n)], 1.0, "units"
    if name == "middlebury":
        items = []
        for disp in sorted(glob.glob(os.path.join(root, "**", "disp1.png"), recursive=True)):
            sd = os.path.dirname(disp)
            views = sorted(glob.glob(os.path.join(sd, "**", "view1.png"), recursive=True))
            views = [v for v in views if "Illum2" in v and "Exp1" in v] or views
            if not views:
                continue
            rgb = cv2.cvtColor(cv2.imread(views[0]), cv2.COLOR_BGR2RGB)
            d = load_depth(disp, "float metres (.npy / .pfm / .tiff)")  # disparity in px
            if rgb.shape[:2] != d.shape:
                rgb = cv2.resize(rgb, d.shape[::-1], interpolation=cv2.INTER_AREA)
            items.append((os.path.basename(sd), *crop16(rgb, d)))
        if limit > 0:
            items = items[:limit]
        if not items:
            raise SystemExit(f"No Middlebury scenes found under {root}")
        return items, 1.0, "disparity px"
    raise SystemExit("unknown dataset")


# ----------------------------------------------------------------------------- memory probe
def peak_memory_mb(method, scale, h, w):
    """Peak extra RSS of one inference in a fresh process (fair per-method memory)."""
    code = f"""
import sys, resource, numpy as np
sys.path.insert(0, {ROOT!r})
from depthsr.hybrid import DepthSR
from depthsr.synthetic import make_scene
from depthsr.degrade import make_lr
sr = DepthSR(device='cpu')
rgb, gt = make_scene({h}, {w}, 1)
lr = make_lr(gt, {scale})
if {method!r} in ('dl','hybrid') and not sr.has_weights: print('nan'); sys.exit()
sr.run(lr[:8,:8], rgb[:8*{scale},:8*{scale}], {scale}, {method!r}) if False else None
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
sr.run(lr, rgb, {scale}, {method!r})
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - base) / 1024)
"""
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        return float(out.stdout.strip().splitlines()[-1])
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- figures
def save_compare(name, rgb, gt, outs, path, unit_mul):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(gt[gt > 0], 1), np.percentile(gt[gt > 0], 99)
    # zoom: window with most GT discontinuities
    _, disc = C.depth_discontinuities(gt, 0.05)
    H, W = gt.shape
    z = min(H, W) // 3
    integ = cv2.boxFilter(disc, -1, (z, z), normalize=False)
    cy, cx = np.unravel_index(np.argmax(integ), integ.shape)
    y0, x0 = int(np.clip(cy - z // 2, 0, H - z)), int(np.clip(cx - z // 2, 0, W - z))
    keys = list(outs.keys())
    cols = len(keys) + 2
    fig, ax = plt.subplots(2, cols, figsize=(2.4 * cols, 4.6))
    panels = [("RGB", rgb), ("GT", colorize(gt, vmin, vmax))] + [(NAMES[k].split(" (")[0], colorize(outs[k], vmin, vmax)) for k in keys]
    for j, (t, im) in enumerate(panels):
        ax[0, j].imshow(im)
        ax[0, j].set_title(t, fontsize=8)
        ax[1, j].imshow(im[y0:y0 + z, x0:x0 + z])
        for a in ax[:, j]:
            a.axis("off")
        if j > 1:
            k = keys[j - 2]
            r = np.sqrt(np.mean((outs[k][gt > 0] - gt[gt > 0]) ** 2)) * unit_mul
            ax[1, j].set_title(f"RMSE {r:.2f}", fontsize=7)
    ax[0, 0].add_patch(plt.Rectangle((x0, y0), z, z, fill=False, color="w", lw=1))
    plt.suptitle(name, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def save_failure(name, rgb, gt, pred, path, unit_mul):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    vmin, vmax = np.percentile(gt[gt > 0], 1), np.percentile(gt[gt > 0], 99)
    err = np.abs(pred - gt) * (gt > 0) * unit_mul
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.8))
    for a, (t, im) in zip(ax, [("RGB", rgb), ("GT", colorize(gt, vmin, vmax)), ("Hybrid", colorize(pred, vmin, vmax))]):
        a.imshow(im)
        a.set_title(t)
        a.axis("off")
    h = ax[3].imshow(err, cmap="inferno", vmax=np.percentile(err, 99.5) + 1e-6)
    ax[3].set_title("|error|")
    ax[3].axis("off")
    plt.colorbar(h, ax=ax[3], fraction=0.046)
    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)


def describe(rgb, gt):
    """Image statistics used to explain failure causes."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    e, _ = C.rgb_edges(rgb)
    _, disc = C.depth_discontinuities(gt, 0.05)
    v = gt[gt > 0]
    return {"brightness": float(gray.mean()), "rgb_edge_density": float((e > 0.3).mean()),
            "depth_disc_density": float(disc.mean()), "depth_range": float(v.max() - v.min()),
            "near_far_ratio": float(np.percentile(v, 99) / max(np.percentile(v, 1), 1e-3))}


def guess_cause(st, med):
    causes = []
    if st["depth_disc_density"] > 1.5 * med["depth_disc_density"]:
        causes.append("many depth discontinuities / thin structures (chair legs, plants, cables) that "
                      "disappear completely in the low-resolution input")
    if st["rgb_edge_density"] > 1.5 * med["rgb_edge_density"]:
        causes.append("heavy RGB texture (shelves, books, patterned surfaces): colour guidance becomes ambiguous")
    if st["brightness"] < 0.7 * med["brightness"]:
        causes.append("dark / low-contrast RGB so object boundaries are not visible in the guide image")
    if st["near_far_ratio"] > 1.5 * med["near_far_ratio"]:
        causes.append("very large depth range (window / corridor): far values have large absolute errors")
    if not causes:
        causes.append("boundary between objects with similar colour (no RGB edge to guide the depth edge), "
                      "or reflective / transparent surfaces where the GT depth itself is unreliable")
    return causes


# ----------------------------------------------------------------------------- main
def fmt(v, k):
    if v != v:
        return "-"
    return f"{v:.4f}" if k in ("SSIM", "EdgeF1") else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyu", choices=["nyu", "middlebury", "synthetic"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "tgnet.pt"))
    ap.add_argument("--scales", default="2,4,8,16")
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    ap.add_argument("--noisy_limit", type=int, default=100, help="images for the noisy+holes protocol")
    ap.add_argument("--team", default="T01")
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--no_memory", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)
    scales = [int(s) for s in args.scales.split(",")]
    items, unit_mul, unit = load_dataset(args.dataset, args.data, args.limit)
    sr = DepthSR(args.ckpt)
    methods = [m for m in METHODS if sr.has_weights or m not in ("dl", "hybrid")]
    print(f"{len(items)} images, methods={methods}, device={sr.device}, weights={sr.has_weights}")

    protocols = {"clean": dict(noise=0.0, holes=0.0)}
    protocols["noisy"] = dict(noise=0.02, holes=0.04)
    res = {"dataset": args.dataset, "unit": unit, "n_images": len(items), "device": sr.device, "results": {}}
    per_image_hybrid = {}
    for pname, deg in protocols.items():
        p_items = items if pname == "clean" else items[: args.noisy_limit]
        p_scales = scales if pname == "clean" else [s for s in scales if s in (4, 8)]
        for s in p_scales:
            acc = {m: {k: [] for k in METRIC_KEYS + ["ms"]} for m in methods}
            for idx, (name, rgb, gt) in enumerate(p_items):
                gt_in = C.fill_holes(gt) if (gt <= 0).any() else gt
                lr = make_lr(gt_in, s, seed=idx, **deg)
                outs = {}
                for m in methods:
                    out, _, ms = sr.run(lr, rgb, s, m, return_aux=True)
                    r = all_metrics(out, gt, rgb)
                    for k in ("RMSE", "MAE", "BoundaryRMSE"):
                        r[k] *= unit_mul
                    for k in METRIC_KEYS:
                        acc[m][k].append(r[k])
                    acc[m]["ms"].append(ms)
                    outs[m] = out
                    if m == "hybrid" and pname == "clean":
                        per_image_hybrid.setdefault(s, []).append((name, r["RMSE"], r["BoundaryRMSE"]))
                if idx < 3 and pname == "clean" and s in (4, 8):
                    show = {k: outs[k] for k in ("bicubic", "jbu", "classical", "dl", "hybrid") if k in outs}
                    save_compare(f"{name} x{s}", rgb, gt, show,
                                 os.path.join(args.out, "figs", f"compare_{args.dataset}_{name}_x{s}.png"), unit_mul)
                if idx % 50 == 0:
                    print(f"[{pname} x{s}] {idx}/{len(p_items)}", flush=True)
            res["results"].setdefault(pname, {})[str(s)] = {
                m: {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in d.items()} for m, d in acc.items()}
            print(pname, s, {m: round(res["results"][pname][str(s)][m]["RMSE"], 3) for m in methods})

    # ---- ablation (hybrid variants) on 4x and 8x, both protocols
    abl_cfgs = {"Full hybrid": {}, "- trust map / gating": {"no_trust": True}, "- hole filling": {"no_fill": True},
                "+ boundary snapping (v1 design)": {"snap": True},
                "- noise-adaptive smoothing": {"no_adapt": True}, "- fusion gate (alpha=1)": {"no_gate": True}}
    ablation = {}
    if sr.has_weights:
        ab_items = items[: min(len(items), args.noisy_limit)]
        for pname, deg in protocols.items():
            for s in (4, 8):
                if s not in scales:
                    continue
                for cname, cfg in abl_cfgs.items():
                    vals = {k: [] for k in ("RMSE", "BoundaryRMSE", "EdgeF1", "TextureCopy")}
                    for idx, (name, rgb, gt) in enumerate(ab_items):
                        gt_in = C.fill_holes(gt) if (gt <= 0).any() else gt
                        lr = make_lr(gt_in, s, seed=idx, **deg)
                        out = sr.run(lr, rgb, s, "hybrid", ablate=cfg)
                        r = all_metrics(out, gt, rgb)
                        for k in vals:
                            vals[k].append(r[k] * (unit_mul if k in ("RMSE", "BoundaryRMSE") else 1))
                    ablation.setdefault(f"{pname} x{s}", {})[cname] = {k: float(np.nanmean(v)) for k, v in vals.items()}
                for ref in ("classical", "dl"):
                    r = res["results"][pname][str(s)][ref]
                    ablation[f"{pname} x{s}"][f"{NAMES[ref]}"] = {k: r[k] for k in ("RMSE", "BoundaryRMSE", "EdgeF1", "TextureCopy")}
        res["ablation"] = ablation

    # ---- memory
    mem = {}
    if not args.no_memory:
        H, W = items[0][2].shape
        for m in methods:
            mem[m] = peak_memory_mb(m, 4, H, W)
        res["memory_mb_x4"] = mem

    # ---- failure cases
    failures = []
    if sr.has_weights and per_image_hybrid:
        s = 8 if 8 in per_image_hybrid else max(per_image_hybrid)
        ranked = sorted(per_image_hybrid[s], key=lambda t: -t[1])[:3]
        lookup = {n: (r, g) for n, r, g in items}
        stats = [describe(r, g) for _, r, g in items[:200]]
        med = {k: float(np.median([st[k] for st in stats])) for k in stats[0]}
        for name, rm, brm in ranked:
            rgb, gt = lookup[name]
            idx = [n for n, _, _ in items].index(name)
            lr = make_lr(C.fill_holes(gt) if (gt <= 0).any() else gt, s, seed=idx)
            pred = sr.run(lr, rgb, s, "hybrid")
            fig = os.path.join(args.out, "figs", f"failure_{args.dataset}_{name}_x{s}.png")
            save_failure(f"{name} x{s}  RMSE {rm:.2f} {unit}", rgb, gt, pred, fig, unit_mul)
            st = describe(rgb, gt)
            failures.append({"image": name, "scale": s, "RMSE": rm, "BoundaryRMSE": brm, "stats": st,
                             "likely_causes": guess_cause(st, med), "figure": os.path.relpath(fig, args.out)})
        res["failure_cases"] = failures

    with open(os.path.join(args.out, f"metrics_{args.dataset}.json"), "w") as f:
        json.dump(res, f, indent=2)
    write_reports(res, args, methods, unit, mem, ablation, failures)
    print("done ->", args.out)


def table(rows_by_method, methods, keys, unit):
    hdr = "| Method | Type | " + " | ".join(f"{k} ({unit})" if k in ("RMSE", "MAE", "BoundaryRMSE") else k for k in keys) + " |"
    out = [hdr, "|" + "---|" * (len(keys) + 2)]
    for m in methods:
        r = rows_by_method[m]
        out.append(f"| {NAMES[m]} | {GROUP[m]} | " + " | ".join(fmt(r[k], k) for k in keys) + " |")
    return "\n".join(out)


def write_reports(res, args, methods, unit, mem, ablation, failures):
    R = res["results"]
    L = [args.usns, "", f"# Evaluation - P4 Edge-Preserving Depth Map Upsampling ({args.team})", "",
         f"Dataset: **{args.dataset}** ({res['n_images']} unseen test images), device: {res['device']}. "
         "Degradation: bicubic downsampling (clean); + 2 % depth-relative Gaussian noise and 4 % random holes (noisy).",
         "PSNR/SSIM on depth normalised by GT max. BoundaryRMSE = RMSE within 5 px of GT depth discontinuities. "
         "EdgeF1 = F1 of depth-edge maps (2 px tolerance). TextureCopy = spurious depth gradient on flat-but-textured "
         "regions (lower is better).", ""]
    first = "4" if "4" in R["clean"] else list(R["clean"])[0]
    L += ["## 1. Depth accuracy - RMSE and MAE", ""]
    for s in R["clean"]:
        L += [f"**x{s}**", "", table(R["clean"][s], methods, ["RMSE", "MAE"], unit), ""]
    L += ["## 2. PSNR and SSIM", ""]
    for s in R["clean"]:
        L += [f"**x{s}**", "", table(R["clean"][s], methods, ["PSNR", "SSIM"], unit), ""]
    L += ["## 3. Edge preservation and boundary accuracy", ""]
    for s in R["clean"]:
        L += [f"**x{s}**", "", table(R["clean"][s], methods, ["BoundaryRMSE", "EdgeF1", "TextureCopy"], unit), ""]
    L += ["## 4. Performance across upsampling factors (RMSE, " + unit + ")", "",
          "| Method | " + " | ".join(f"x{s}" for s in R["clean"]) + " |", "|---|" + "---|" * len(R["clean"])]
    for m in methods:
        L.append(f"| {NAMES[m]} | " + " | ".join(fmt(R["clean"][s][m]["RMSE"], "RMSE") for s in R["clean"]) + " |")
    if "noisy" in R:
        L += ["", "**Noisy + missing depth (robustness)**", ""]
        for s in R["noisy"]:
            L += [f"x{s}", "", table(R["noisy"][s], methods, ["RMSE", "MAE", "BoundaryRMSE", "EdgeF1"], unit), ""]
    L += ["", f"## 5. Processing latency and memory (x{first}, {res['device']})", "",
          "| Method | Latency (ms / image) | Peak extra memory (MB, CPU) |", "|---|---|---|"]
    for m in methods:
        L.append(f"| {NAMES[m]} | {R['clean'][first][m]['ms']:.1f} | {mem.get(m, float('nan')):.1f} |")
    L += ["", "Model: TGNet, 1.99 M parameters, ~7.6 MB (fp32)."]
    with open(os.path.join(args.out, f"eval_P4_{args.team}.md" if args.dataset == "nyu" else f"eval_{args.dataset}.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    if ablation:
        A = [f"# Ablation study ({args.dataset})", "",
             "Inference-time ablation: each component of the hybrid pipeline is switched off individually "
             "with the same trained weights; classical-only and DL-only rows show each half alone.", ""]
        for setting, rows in ablation.items():
            A += [f"## {setting}", "", f"| Variant | RMSE ({unit}) | Boundary RMSE ({unit}) | Edge F1 | Texture copy |",
                  "|---|---|---|---|---|"]
            for v, r in rows.items():
                A.append(f"| {v} | {fmt(r['RMSE'],'RMSE')} | {fmt(r['BoundaryRMSE'],'x')} | {fmt(r['EdgeF1'],'EdgeF1')} | {fmt(r['TextureCopy'],'x')} |")
            A.append("")
        with open(os.path.join(args.out, f"ablation_{args.dataset}.md"), "w") as f:
            f.write("\n".join(A))
    if failures:
        F_ = [f"# Failure cases ({args.dataset})", "",
              "The three test images with the highest hybrid RMSE. Causes are inferred from image statistics "
              "compared with the dataset median (see describe()/guess_cause() in scripts/evaluate.py) and should be "
              "confirmed by looking at the figures.", ""]
        for i, fc in enumerate(failures, 1):
            F_ += [f"## {i}. {fc['image']} (x{fc['scale']}) - RMSE {fc['RMSE']:.2f} {unit}, boundary RMSE {fc['BoundaryRMSE']:.2f}",
                   "", f"![]({fc['figure']})", "", "Likely causes:"] + [f"- {c}" for c in fc["likely_causes"]] + [
                   "", "Stats: " + ", ".join(f"{k}={v:.3f}" for k, v in fc["stats"].items()), ""]
        with open(os.path.join(args.out, f"failure_cases_{args.dataset}.md"), "w") as f:
            f.write("\n".join(F_))


if __name__ == "__main__":
    main()
