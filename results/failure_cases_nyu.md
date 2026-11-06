# Failure cases (nyu)

The three test images with the highest hybrid RMSE. Causes are inferred from image statistics compared with the dataset median (see describe()/guess_cause() in scripts/evaluate.py) and should be confirmed by looking at the figures.

## 1. nyu_1015 (x8) - RMSE 24.92 cm, boundary RMSE 60.84

![](figs/failure_nyu_nyu_1015_x8.png)

Likely causes:
- many depth discontinuities / thin structures (chair legs, plants, cables) that disappear completely in the low-resolution input
- dark / low-contrast RGB so object boundaries are not visible in the guide image

Stats: brightness=57.815, rgb_edge_density=0.124, depth_disc_density=0.045, depth_range=6.307, near_far_ratio=5.007

## 2. nyu_1025 (x8) - RMSE 10.68 cm, boundary RMSE 39.52

![](figs/failure_nyu_nyu_1025_x8.png)

Likely causes:
- boundary between objects with similar colour (no RGB edge to guide the depth edge), or reflective / transparent surfaces where the GT depth itself is unreliable

Stats: brightness=92.769, rgb_edge_density=0.090, depth_disc_density=0.013, depth_range=7.803, near_far_ratio=2.940

## 3. nyu_1014 (x8) - RMSE 9.71 cm, boundary RMSE 28.44

![](figs/failure_nyu_nyu_1014_x8.png)

Likely causes:
- many depth discontinuities / thin structures (chair legs, plants, cables) that disappear completely in the low-resolution input
- dark / low-contrast RGB so object boundaries are not visible in the guide image
- very large depth range (window / corridor): far values have large absolute errors

Stats: brightness=53.003, rgb_edge_density=0.113, depth_disc_density=0.021, depth_range=8.467, near_far_ratio=5.576
