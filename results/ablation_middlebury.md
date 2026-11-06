# Ablation study (middlebury)

Inference-time ablation: each component of the hybrid pipeline is switched off individually with the same trained weights; classical-only and DL-only rows show each half alone.

## clean x4

| Variant | RMSE (disparity px) | Boundary RMSE (disparity px) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 2.689 | 6.441 | 0.8660 | 0.693 |
| - trust map / gating | 2.705 | 6.455 | 0.8536 | 0.685 |
| - hole filling | 2.703 | 6.478 | 0.8657 | 0.716 |
| + boundary snapping (v1 design) | 3.206 | 7.707 | 0.8679 | 0.868 |
| - noise-adaptive smoothing | 2.720 | 6.491 | 0.8536 | 0.664 |
| - fusion gate (alpha=1) | 2.685 | 6.427 | 0.8702 | 0.679 |
| Classical pipeline (ours) | 3.116 | 7.529 | 0.7978 | 0.964 |
| Deep learning only (TGNet) | 3.056 | 7.397 | 0.8498 | 0.745 |

## clean x8

| Variant | RMSE (disparity px) | Boundary RMSE (disparity px) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 4.020 | 9.585 | 0.7884 | 0.864 |
| - trust map / gating | 3.924 | 9.241 | 0.7884 | 0.891 |
| - hole filling | 4.092 | 9.776 | 0.7880 | 0.897 |
| + boundary snapping (v1 design) | 4.349 | 10.375 | 0.7774 | 1.233 |
| - noise-adaptive smoothing | 4.002 | 9.447 | 0.7775 | 0.881 |
| - fusion gate (alpha=1) | 4.045 | 9.635 | 0.7985 | 0.872 |
| Classical pipeline (ours) | 4.756 | 11.380 | 0.6189 | 1.069 |
| Deep learning only (TGNet) | 4.412 | 10.606 | 0.7577 | 0.842 |

## noisy x4

| Variant | RMSE (disparity px) | Boundary RMSE (disparity px) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 3.524 | 8.088 | 0.8228 | 0.717 |
| - trust map / gating | 3.575 | 8.050 | 0.8225 | 0.824 |
| - hole filling | 23.814 | 24.795 | 0.4748 | 4.228 |
| + boundary snapping (v1 design) | 3.749 | 8.660 | 0.8322 | 0.787 |
| - noise-adaptive smoothing | 3.524 | 8.088 | 0.8228 | 0.717 |
| - fusion gate (alpha=1) | 3.534 | 8.105 | 0.8261 | 0.707 |
| Classical pipeline (ours) | 3.710 | 8.515 | 0.7975 | 0.944 |
| Deep learning only (TGNet) | 10.791 | 12.295 | 0.7122 | 0.973 |

## noisy x8

| Variant | RMSE (disparity px) | Boundary RMSE (disparity px) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 4.496 | 10.333 | 0.7455 | 0.914 |
| - trust map / gating | 4.462 | 10.157 | 0.7606 | 0.950 |
| - hole filling | 22.897 | 24.408 | 0.4352 | 3.559 |
| + boundary snapping (v1 design) | 4.592 | 10.532 | 0.7463 | 1.088 |
| - noise-adaptive smoothing | 4.496 | 10.333 | 0.7455 | 0.914 |
| - fusion gate (alpha=1) | 4.528 | 10.402 | 0.7541 | 0.933 |
| Classical pipeline (ours) | 4.802 | 10.948 | 0.6763 | 1.056 |
| Deep learning only (TGNet) | 11.005 | 14.171 | 0.6330 | 1.124 |
