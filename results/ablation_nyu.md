# Ablation study (nyu)

Inference-time ablation: each component of the hybrid pipeline is switched off individually with the same trained weights; classical-only and DL-only rows show each half alone.

## clean x4

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 3.147 | 10.526 | 0.9763 | 0.267 |
| - trust map / gating | 4.017 | 13.301 | 0.9279 | 0.316 |
| - hole filling | 3.147 | 10.526 | 0.9763 | 0.267 |
| + boundary snapping (v1 design) | 4.952 | 17.070 | 0.9727 | 0.275 |
| - noise-adaptive smoothing | 3.750 | 12.045 | 0.9519 | 0.252 |
| - fusion gate (alpha=1) | 3.188 | 10.653 | 0.9779 | 0.263 |
| Classical pipeline (ours) | 3.057 | 10.615 | 0.8613 | 0.348 |
| Deep learning only (TGNet) | 2.475 | 8.349 | 0.9571 | 0.262 |

## clean x8

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 5.617 | 18.486 | 0.9040 | 0.348 |
| - trust map / gating | 6.482 | 21.203 | 0.8466 | 0.449 |
| - hole filling | 5.617 | 18.486 | 0.9040 | 0.348 |
| + boundary snapping (v1 design) | 6.638 | 22.090 | 0.9090 | 0.372 |
| - noise-adaptive smoothing | 6.095 | 19.648 | 0.8675 | 0.346 |
| - fusion gate (alpha=1) | 5.714 | 18.792 | 0.9114 | 0.346 |
| Classical pipeline (ours) | 5.528 | 18.709 | 0.6285 | 0.438 |
| Deep learning only (TGNet) | 4.378 | 14.556 | 0.8741 | 0.330 |

## noisy x4

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 5.456 | 15.880 | 0.9308 | 0.297 |
| - trust map / gating | 5.940 | 16.948 | 0.8966 | 0.458 |
| - hole filling | 54.384 | 49.004 | 0.3147 | 3.551 |
| + boundary snapping (v1 design) | 6.442 | 19.887 | 0.9295 | 0.311 |
| - noise-adaptive smoothing | 5.456 | 15.880 | 0.9308 | 0.297 |
| - fusion gate (alpha=1) | 5.480 | 15.924 | 0.9357 | 0.292 |
| Classical pipeline (ours) | 5.997 | 17.748 | 0.8537 | 0.429 |
| Deep learning only (TGNet) | 22.875 | 24.508 | 0.6973 | 0.505 |

## noisy x8

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 7.370 | 22.359 | 0.8337 | 0.407 |
| - trust map / gating | 7.764 | 23.610 | 0.8137 | 0.545 |
| - hole filling | 54.818 | 57.462 | 0.2793 | 2.355 |
| + boundary snapping (v1 design) | 7.936 | 24.573 | 0.8296 | 0.435 |
| - noise-adaptive smoothing | 7.370 | 22.359 | 0.8337 | 0.407 |
| - fusion gate (alpha=1) | 7.425 | 22.521 | 0.8422 | 0.407 |
| Classical pipeline (ours) | 8.281 | 25.267 | 0.6800 | 0.486 |
| Deep learning only (TGNet) | 23.734 | 30.884 | 0.6350 | 0.579 |
