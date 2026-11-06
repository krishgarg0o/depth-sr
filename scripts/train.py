"""Train TGNet.

  python scripts/train.py --data data --iters 15000 --bs 16          # NYU (Kaggle T4: ~40 min)
  python scripts/train.py --synthetic --iters 300 --bs 4              # quick CPU smoke test

Loss = L1 + 0.5 * gradient L1 + boundary-weighted L1 (5x weight near GT discontinuities)
Best checkpoint (lowest val RMSE) -> checkpoints/tgnet.pt ; log -> results/train_log.csv
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depthsr.data import DepthSRDataset  # noqa: E402
from depthsr.model import TGNet, count_params  # noqa: E402


def grad(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def loss_fn(pred, gt):
    m = (gt > 0).float()
    gx, gy = grad(gt)
    edge = ((gx.abs() > 0.03).float()[..., :-1, :] + (gy.abs() > 0.03).float()[..., :, :-1]).clamp(0, 1)
    edge = F.max_pool2d(F.pad(edge, (0, 1, 0, 1)), 5, 1, 2)
    w = m * (1 + 4 * edge)
    l1 = (w * (pred - gt).abs()).sum() / w.sum()
    px, py = grad(pred)
    lg = (px - gx).abs().mean() + (py - gy).abs().mean()
    return l1 + 0.5 * lg


def batch_to(b, dev):
    return [b[k].to(dev, non_blocking=True) for k in ("init", "valid", "scale_plane", "rgb", "trust", "band")], b["gt"].to(dev)


@torch.no_grad()
def validate(net, loader, dev):
    net.eval()
    se, se0, n = 0.0, 0.0, 0
    for b in loader:
        args, gt = batch_to(b, dev)
        pred = net(*args)
        se += ((pred - gt) ** 2).mean().item()
        se0 += ((args[0] - gt) ** 2).mean().item()   # network input (classical / bicubic) baseline
        n += 1
    net.train()
    return (se / max(n, 1)) ** 0.5, (se0 / max(n, 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--out", default="checkpoints/tgnet.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.synthetic:
        tr = DepthSRDataset.synthetic(48, crop=args.crop, length=args.iters * args.bs)
        va = DepthSRDataset.synthetic(8, crop=args.crop, train=False, length=16)
    else:
        t = np.load(os.path.join(args.data, "nyu_train.npz"))
        v = np.load(os.path.join(args.data, "nyu_val.npz"))
        tr = DepthSRDataset(t["rgb"], t["depth"], crop=args.crop, length=args.iters * args.bs)
        va = DepthSRDataset(v["rgb"], v["depth"], crop=args.crop, train=False, length=len(v["rgb"]) * 2)
    tl = DataLoader(tr, args.bs, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va, args.bs, num_workers=args.workers)

    net = TGNet().to(dev)
    print(f"device={dev} params={count_params(net)/1e6:.2f}M train_samples={len(tr)}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log = open("results/train_log.csv", "w", newline="")
    wr = csv.writer(log)
    wr.writerow(["iter", "train_loss", "val_rmse_norm", "val_rmse_input_baseline", "sec"])
    best, t0, run = 1e9, time.time(), 0.0
    for it, b in enumerate(tl, 1):
        a, gt = batch_to(b, dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            pred = net(*a)
        loss = loss_fn(pred.float(), gt)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it > 1 else loss.item()
        if it % 50 == 0:
            print(f"it {it}/{args.iters} loss {run:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if it % args.val_every == 0 or it == args.iters:
            v, v0 = validate(net, vl, dev)
            wr.writerow([it, f"{run:.5f}", f"{v:.5f}", f"{v0:.5f}", f"{time.time()-t0:.0f}"])
            log.flush()
            print(f"  val rmse(norm) net {v:.5f}  vs input baseline {v0:.5f}  "
                  f"({(1 - v / v0) * 100:+.1f}% better)", flush=True)
            if v < best:
                best = v
                torch.save({"model": net.state_dict(), "iter": it, "val": v}, args.out)
                print("  saved", args.out)
        if it >= args.iters:
            break
    print("done. best val", best)


if __name__ == "__main__":
    main()
