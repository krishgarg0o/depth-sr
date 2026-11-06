# Ablation study (nyu)

Inference-time ablation: each component of the hybrid pipeline is switched off individually with the same trained weights; classical-only and DL-only rows show each half alone.

## clean x4

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 3.613 | 11.994 | 0.9700 | 0.266 |
| - trust map / gating | 4.616 | 15.558 | 0.9499 | 0.313 |
| - hole filling | 3.613 | 11.994 | 0.9700 | 0.266 |
| - boundary snapping | 3.582 | 11.848 | 0.9514 | 0.282 |
| - fusion gate (alpha=1) | 3.629 | 12.021 | 0.9689 | 0.272 |
| Classical pipeline (ours) | 4.721 | 16.481 | 0.9469 | 0.282 |
| Deep learning only (TGNet) | 2.503 | 8.451 | 0.9634 | 0.268 |

## clean x8

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 5.844 | 19.140 | 0.8918 | 0.356 |
| - trust map / gating | 6.511 | 21.409 | 0.8485 | 0.437 |
| - hole filling | 5.844 | 19.140 | 0.8918 | 0.356 |
| - boundary snapping | 5.719 | 18.642 | 0.8884 | 0.357 |
| - fusion gate (alpha=1) | 5.940 | 19.389 | 0.8986 | 0.364 |
| Classical pipeline (ours) | 6.444 | 21.825 | 0.7847 | 0.420 |
| Deep learning only (TGNet) | 4.363 | 14.590 | 0.8757 | 0.336 |

## noisy x4

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 5.485 | 15.921 | 0.9373 | 0.299 |
| - trust map / gating | 6.446 | 18.956 | 0.9155 | 0.442 |
| - hole filling | 55.114 | 49.635 | 0.3394 | 3.268 |
| - boundary snapping | 5.402 | 15.608 | 0.9223 | 0.313 |
| - fusion gate (alpha=1) | 5.506 | 15.925 | 0.9383 | 0.299 |
| Classical pipeline (ours) | 7.200 | 22.685 | 0.9135 | 0.411 |
| Deep learning only (TGNet) | 22.878 | 24.585 | 0.6949 | 0.479 |

## noisy x8

| Variant | RMSE (cm) | Boundary RMSE (cm) | Edge F1 | Texture copy |
|---|---|---|---|---|
| Full hybrid | 7.258 | 21.867 | 0.8397 | 0.406 |
| - trust map / gating | 7.932 | 23.857 | 0.8134 | 0.534 |
| - hole filling | 55.725 | 58.027 | 0.2967 | 2.148 |
| - boundary snapping | 7.190 | 21.560 | 0.8374 | 0.405 |
| - fusion gate (alpha=1) | 7.315 | 21.999 | 0.8475 | 0.413 |
| Classical pipeline (ours) | 9.073 | 28.440 | 0.7497 | 0.503 |
| Deep learning only (TGNet) | 23.715 | 30.818 | 0.6335 | 0.531 |
