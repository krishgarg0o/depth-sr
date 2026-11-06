"""Training dataset: random crops of NYU (or synthetic) with on-the-fly degradation.

Each sample randomly picks:
  * scale  in {2, 4, 8, 16}
  * noise  (50 %: relative sigma U(0, 0.03))
  * holes  (30 %: U(0, 0.06) of the LR image)
  * mode   hybrid (75 %) or dl-only (25 %) -> one network learns both settings,
           which gives the "DL only" baseline without training a second model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import make_lr
from .hybrid import build_inputs
from .synthetic import make_scene


class DepthSRDataset(Dataset):
    def __init__(self, rgb, depth, crop=256, scales=(2, 4, 8, 16), length=None, train=True,
                 p_dl=0.25, seed=0):
        self.rgb, self.depth = rgb, depth
        self.crop, self.scales, self.train, self.p_dl = crop, scales, train, p_dl
        self.length = length or len(rgb)
        self.seed = seed

    @classmethod
    def synthetic(cls, n=64, **kw):
        scenes = [make_scene(288, 352, seed=i) for i in range(n)]
        return cls(np.stack([s[0] for s in scenes]), np.stack([s[1] for s in scenes]), **kw)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.train else self.seed + idx)
        i = int(rng.integers(len(self.rgb))) if self.train else idx % len(self.rgb)
        rgb, gt = self.rgb[i], self.depth[i]
        H, W = gt.shape
        c = self.crop
        y, x = int(rng.integers(0, H - c + 1)), int(rng.integers(0, W - c + 1))
        rgb, gt = rgb[y:y + c, x:x + c], gt[y:y + c, x:x + c]
        if self.train and rng.random() < 0.5:
            rgb, gt = rgb[:, ::-1], gt[:, ::-1]
        rgb, gt = np.ascontiguousarray(rgb), np.ascontiguousarray(gt).astype(np.float32)
        s = int(rng.choice(self.scales))
        noise = float(rng.uniform(0, 0.03)) if rng.random() < 0.5 else 0.0
        holes = float(rng.uniform(0, 0.06)) if rng.random() < 0.3 else 0.0
        lr = make_lr(gt, s, noise, holes, seed=int(rng.integers(1 << 30)))
        mode = "dl" if rng.random() < self.p_dl else "hybrid"
        inp, _ = build_inputs(lr, rgb, s, mode)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        sample = {k: t(inp[k])[None] if inp[k].ndim == 2 else t(inp[k])
                  for k in ("init", "valid", "scale_plane", "rgb", "trust", "band")}
        sample["gt"] = t(gt / inp["norm"])[None]
        sample["scale"] = s
        return sample
