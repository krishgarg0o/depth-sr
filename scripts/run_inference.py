"""Batch inference from the command line.

Input folder layout (any number of scenes):
    <name>_rgb.png            high-resolution RGB guide
    <name>_depth.png          depth map (16-bit PNG in mm, KITTI PNG, 8-bit PNG, .npy, .pfm or .tiff)

Two modes:
    --mode simulate   (default) <name>_depth is full resolution: it is downsampled by --scale
                      (plus optional --noise / --holes), reconstructed, and scored against itself.
    --mode upsample   <name>_depth is a real low-resolution map: it is upsampled to the RGB size.

Examples:
    python scripts/run_inference.py --input_dir ./samples/ --output_dir ./results/inference/
    python scripts/run_inference.py --input_dir ./samples/ --output_dir ./out/ --scale 4 --noise 0.02 --holes 0.04
    python scripts/run_inference.py --input_dir ./my_lowres/ --output_dir ./out/ --mode upsample --scale 8

Outputs per scene: <name>_<method>_mm.png (16-bit, millimetres), <name>_<method>_vis.png (colour map),
plus metrics.csv (simulate mode) with RMSE / MAE / PSNR / SSIM / boundary RMSE / edge F1 / latency.
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from depthsr import classical as C  # noqa: E402
from depthsr.degrade import make_lr  # noqa: E402
from depthsr.hybrid import METHODS, DepthSR  # noqa: E402
from depthsr.io_utils import check_pair, colorize, load_depth, load_rgb  # noqa: E402
from depthsr.metrics import all_metrics  # noqa: E402


def find_scenes(folder):
    scenes = []
    for rgb in sorted(glob.glob(os.path.join(folder, "*_rgb.png"))):
        stem = rgb[: -len("_rgb.png")]
        for ext in (".png", ".npy", ".pfm", ".tif", ".tiff"):
            if os.path.exists(stem + "_depth" + ext):
                scenes.append((os.path.basename(stem), rgb, stem + "_depth" + ext))
                break
    return scenes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--mode", choices=["simulate", "upsample"], default="simulate")
    ap.add_argument("--scale", type=int, default=8, choices=[2, 4, 8, 16])
    ap.add_argument("--noise", type=float, default=0.0, help="simulate mode: relative depth noise sigma")
    ap.add_argument("--holes", type=float, default=0.0, help="simulate mode: fraction of missing depth")
    ap.add_argument("--methods", default="bicubic,classical,hybrid",
                    help="comma list from: " + ",".join(METHODS))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "tgnet.pt"))
    args = ap.parse_args()

    scenes = find_scenes(args.input_dir)
    if not scenes:
        raise SystemExit(f"No '<name>_rgb.png' + '<name>_depth.*' pairs found in {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    sr = DepthSR(args.ckpt)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m}")
        if m in ("dl", "hybrid") and not sr.has_weights:
            raise SystemExit(f"{m} needs trained weights at {args.ckpt}")
    print(f"{len(scenes)} scenes, methods={methods}, mode={args.mode}, x{args.scale}, device={sr.device}")

    rows = []
    for name, rgb_path, depth_path in scenes:
        rgb = load_rgb(rgb_path)
        depth = load_depth(depth_path)
        gt = None
        if args.mode == "simulate":
            if depth.shape != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            H, W = depth.shape
            H2, W2 = H // args.scale * args.scale, W // args.scale * args.scale
            gt, rgb = depth[:H2, :W2], rgb[:H2, :W2]
            gt_in = C.fill_holes(gt) if (gt <= 0).any() else gt
            lr = make_lr(gt_in, args.scale, args.noise, args.holes, seed=0)
        else:
            lr = depth
            rgb = check_pair(rgb, lr, args.scale)
        vmin, vmax = None, None
        if gt is not None:
            vmin, vmax = np.percentile(gt[gt > 0], 1), np.percentile(gt[gt > 0], 99)
        cache = {}
        for m in methods:
            out, _, ms = sr.run(lr, rgb, args.scale, m, return_aux=True, cache=cache)
            base = os.path.join(args.output_dir, f"{name}_{m}")
            cv2.imwrite(base + "_mm.png", np.clip(out * 1000, 0, 65535).astype(np.uint16))
            cv2.imwrite(base + "_vis.png", cv2.cvtColor(colorize(out, vmin, vmax), cv2.COLOR_RGB2BGR))
            row = {"scene": name, "method": m, "latency_ms": round(ms, 1)}
            if gt is not None:
                r = all_metrics(out, gt, rgb)
                row.update({k: round(v, 5) for k, v in r.items()})
            rows.append(row)
            print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    if args.mode == "simulate":
        path = os.path.join(args.output_dir, "metrics.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("metrics ->", path)
    print("outputs ->", args.output_dir)


if __name__ == "__main__":
    main()
