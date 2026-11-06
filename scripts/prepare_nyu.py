"""Convert nyu_depth_v2_labeled.mat -> data/nyu_{train,val,test}.npz

Split (standard guided depth SR protocol, e.g. DKN / DCTNet / FDSR):
  images 0-999  -> train (0-899) + val (900-999)
  images 1000-1448 -> test (449 images, never seen in training)
An 8 px border is cropped (NYU has white/invalid borders) -> 464 x 624 (divisible by 16).

Usage:
  python scripts/prepare_nyu.py --mat nyu_depth_v2_labeled.mat --out data
"""
import argparse
import os

import h5py
import numpy as np

B = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with h5py.File(args.mat, "r") as f:
        imgs = f["images"]    # (N, 3, W, H)
        deps = f["depths"]    # (N, W, H)
        n = imgs.shape[0]
        print("images", imgs.shape, "depths", deps.shape)
        rgb = np.empty((n, 480 - 2 * B, 640 - 2 * B, 3), np.uint8)
        dep = np.empty((n, 480 - 2 * B, 640 - 2 * B), np.float32)
        for i in range(n):
            r = np.asarray(imgs[i]).transpose(2, 1, 0)       # -> H, W, 3
            d = np.asarray(deps[i]).transpose(1, 0)          # -> H, W (metres)
            rgb[i] = r[B:-B, B:-B]
            dep[i] = d[B:-B, B:-B]
            if i % 200 == 0:
                print(i, "/", n)
    splits = {"train": slice(0, 900), "val": slice(900, 1000), "test": slice(1000, n)}
    for name, sl in splits.items():
        p = os.path.join(args.out, f"nyu_{name}.npz")
        np.savez(p, rgb=rgb[sl], depth=dep[sl])
        print("saved", p, rgb[sl].shape)


if __name__ == "__main__":
    main()
