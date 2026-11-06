"""Smoke tests: python -m pytest -q   (or python tests/test_pipeline.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from depthsr import classical as C  # noqa: E402
from depthsr.degrade import make_lr, sparsify  # noqa: E402
from depthsr.hybrid import DepthSR  # noqa: E402
from depthsr.io_utils import InputError, check_pair  # noqa: E402
from depthsr.metrics import all_metrics  # noqa: E402
from depthsr.synthetic import make_scene  # noqa: E402


def test_all_scales_and_methods():
    sr = DepthSR()
    rgb, gt = make_scene(128, 160, seed=1)
    methods = ["nearest", "bilinear", "bicubic", "jbu", "guided", "classical"] + (["dl", "hybrid"] if sr.has_weights else [])
    for s in (2, 4, 8, 16):
        lr = make_lr(gt, s, noise=0.01, holes=0.05, seed=0)
        for m in methods:
            out = sr.run(lr, rgb, s, m)
            assert out.shape == gt.shape and np.isfinite(out).all()


def test_classical_beats_bicubic():
    rgb, gt = make_scene(192, 256, seed=3)
    lr = make_lr(gt, 8)
    bic = C.interpolate(lr, gt.shape, "bicubic")
    ref = C.classical_refine(lr, rgb, 8)
    assert all_metrics(ref, gt)["BoundaryRMSE"] < all_metrics(bic, gt)["BoundaryRMSE"]


def test_sparse_and_holes_filled():
    rgb, gt = make_scene(128, 160, seed=2)
    lr = sparsify(make_lr(gt, 4), 200)
    filled = C.fill_holes(lr)
    assert (filled > 0).all()


def test_bad_pair_rejected():
    try:
        check_pair(np.zeros((100, 400, 3), np.uint8), np.ones((50, 50), np.float32), 2)
        raise AssertionError("should fail")
    except InputError:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
