1RUA24CSE0186,1RUA24CSE0197,1RUA24CSE0208,1RUA24CSE0209

# Evaluation - P4 Edge-Preserving Depth Map Upsampling (T74)

Dataset: **nyu** (100 unseen test images), device: cuda. Degradation: bicubic downsampling (clean); + 2 % depth-relative Gaussian noise and 4 % random holes (noisy).
PSNR/SSIM on depth normalised by GT max. BoundaryRMSE = RMSE within 5 px of GT depth discontinuities. EdgeF1 = F1 of depth-edge maps (2 px tolerance). TextureCopy = spurious depth gradient on flat-but-textured regions (lower is better).

## 1. Depth accuracy - RMSE and MAE

**x2**

| Method | Type | RMSE (cm) | MAE (cm) |
|---|---|---|---|
| Nearest | Interpolation | 2.405 | 0.650 |
| Bilinear | Interpolation | 1.665 | 0.447 |
| Bicubic | Interpolation | 1.539 | 0.444 |
| Joint Bilateral (JBU) | Classical refinement | 1.724 | 0.530 |
| Guided Filter | Classical refinement | 2.910 | 0.999 |
| Classical pipeline (ours) | Classical refinement | 3.623 | 0.920 |
| Deep learning only (TGNet) | Deep learning | 1.248 | 0.409 |
| Hybrid (ours) | Hybrid | 1.805 | 0.562 |

**x4**

| Method | Type | RMSE (cm) | MAE (cm) |
|---|---|---|---|
| Nearest | Interpolation | 4.748 | 1.272 |
| Bilinear | Interpolation | 3.286 | 0.922 |
| Bicubic | Interpolation | 3.230 | 0.976 |
| Joint Bilateral (JBU) | Classical refinement | 3.353 | 1.104 |
| Guided Filter | Classical refinement | 4.932 | 1.869 |
| Classical pipeline (ours) | Classical refinement | 4.721 | 1.398 |
| Deep learning only (TGNet) | Deep learning | 2.503 | 0.760 |
| Hybrid (ours) | Hybrid | 2.785 | 0.924 |

**x8**

| Method | Type | RMSE (cm) | MAE (cm) |
|---|---|---|---|
| Nearest | Interpolation | 8.044 | 2.423 |
| Bilinear | Interpolation | 5.756 | 1.837 |
| Bicubic | Interpolation | 5.820 | 2.006 |
| Joint Bilateral (JBU) | Classical refinement | 5.518 | 2.008 |
| Guided Filter | Classical refinement | 7.274 | 3.027 |
| Classical pipeline (ours) | Classical refinement | 6.444 | 2.277 |
| Deep learning only (TGNet) | Deep learning | 4.363 | 1.428 |
| Hybrid (ours) | Hybrid | 4.469 | 1.576 |

**x16**

| Method | Type | RMSE (cm) | MAE (cm) |
|---|---|---|---|
| Nearest | Interpolation | 12.700 | 4.546 |
| Bilinear | Interpolation | 9.313 | 3.449 |
| Bicubic | Interpolation | 9.555 | 3.779 |
| Joint Bilateral (JBU) | Classical refinement | 8.627 | 3.453 |
| Guided Filter | Classical refinement | 9.101 | 3.815 |
| Classical pipeline (ours) | Classical refinement | 9.074 | 3.601 |
| Deep learning only (TGNet) | Deep learning | 7.023 | 2.575 |
| Hybrid (ours) | Hybrid | 6.881 | 2.551 |

## 2. PSNR and SSIM

**x2**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 46.497 | 0.9940 |
| Bilinear | Interpolation | 49.797 | 0.9972 |
| Bicubic | Interpolation | 50.516 | 0.9975 |
| Joint Bilateral (JBU) | Classical refinement | 49.501 | 0.9967 |
| Guided Filter | Classical refinement | 44.898 | 0.9918 |
| Classical pipeline (ours) | Classical refinement | 43.111 | 0.9915 |
| Deep learning only (TGNet) | Deep learning | 52.438 | 0.9982 |
| Hybrid (ours) | Hybrid | 49.208 | 0.9969 |

**x4**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 40.579 | 0.9791 |
| Bilinear | Interpolation | 43.865 | 0.9909 |
| Bicubic | Interpolation | 44.025 | 0.9905 |
| Joint Bilateral (JBU) | Classical refinement | 43.710 | 0.9898 |
| Guided Filter | Classical refinement | 40.314 | 0.9834 |
| Classical pipeline (ours) | Classical refinement | 40.784 | 0.9855 |
| Deep learning only (TGNet) | Deep learning | 46.293 | 0.9944 |
| Hybrid (ours) | Hybrid | 45.379 | 0.9934 |

**x8**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 35.993 | 0.9554 |
| Bilinear | Interpolation | 38.998 | 0.9805 |
| Bicubic | Interpolation | 38.892 | 0.9787 |
| Joint Bilateral (JBU) | Classical refinement | 39.394 | 0.9797 |
| Guided Filter | Classical refinement | 36.939 | 0.9761 |
| Classical pipeline (ours) | Classical refinement | 38.106 | 0.9773 |
| Deep learning only (TGNet) | Deep learning | 41.488 | 0.9876 |
| Hybrid (ours) | Hybrid | 41.260 | 0.9872 |

**x16**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 31.989 | 0.9327 |
| Bilinear | Interpolation | 34.793 | 0.9708 |
| Bicubic | Interpolation | 34.592 | 0.9685 |
| Joint Bilateral (JBU) | Classical refinement | 35.495 | 0.9712 |
| Guided Filter | Classical refinement | 35.007 | 0.9731 |
| Classical pipeline (ours) | Classical refinement | 35.060 | 0.9699 |
| Deep learning only (TGNet) | Deep learning | 37.391 | 0.9798 |
| Hybrid (ours) | Hybrid | 37.570 | 0.9803 |

## 3. Edge preservation and boundary accuracy

**x2**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 8.292 | 0.9761 | 0.253 |
| Bilinear | Interpolation | 5.785 | 0.9233 | 0.255 |
| Bicubic | Interpolation | 5.338 | 0.9749 | 0.266 |
| Joint Bilateral (JBU) | Classical refinement | 5.864 | 0.9515 | 0.276 |
| Guided Filter | Classical refinement | 9.906 | 0.8567 | 0.360 |
| Classical pipeline (ours) | Classical refinement | 12.882 | 0.9649 | 0.225 |
| Deep learning only (TGNet) | Deep learning | 4.128 | 0.9894 | 0.242 |
| Hybrid (ours) | Hybrid | 6.128 | 0.9815 | 0.227 |

**x4**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 16.376 | 0.9598 | 0.310 |
| Bilinear | Interpolation | 11.488 | 0.7578 | 0.303 |
| Bicubic | Interpolation | 11.276 | 0.8238 | 0.351 |
| Joint Bilateral (JBU) | Classical refinement | 11.409 | 0.8823 | 0.392 |
| Guided Filter | Classical refinement | 16.525 | 0.7028 | 0.502 |
| Classical pipeline (ours) | Classical refinement | 16.481 | 0.9469 | 0.282 |
| Deep learning only (TGNet) | Deep learning | 8.451 | 0.9634 | 0.268 |
| Hybrid (ours) | Hybrid | 9.359 | 0.9631 | 0.267 |

**x8**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 27.518 | 0.7364 | 0.386 |
| Bilinear | Interpolation | 19.989 | 0.4246 | 0.357 |
| Bicubic | Interpolation | 19.812 | 0.4822 | 0.424 |
| Joint Bilateral (JBU) | Classical refinement | 18.443 | 0.7611 | 0.564 |
| Guided Filter | Classical refinement | 23.204 | 0.5419 | 0.618 |
| Classical pipeline (ours) | Classical refinement | 21.825 | 0.7847 | 0.420 |
| Deep learning only (TGNet) | Deep learning | 14.590 | 0.8757 | 0.336 |
| Hybrid (ours) | Hybrid | 14.782 | 0.8891 | 0.344 |

**x16**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 41.942 | 0.3632 | 0.410 |
| Bilinear | Interpolation | 30.456 | 0.2043 | 0.353 |
| Bicubic | Interpolation | 30.592 | 0.2601 | 0.420 |
| Joint Bilateral (JBU) | Classical refinement | 27.292 | 0.5607 | 0.636 |
| Guided Filter | Classical refinement | 28.328 | 0.3767 | 0.551 |
| Classical pipeline (ours) | Classical refinement | 28.885 | 0.5079 | 0.490 |
| Deep learning only (TGNet) | Deep learning | 22.583 | 0.7196 | 0.432 |
| Hybrid (ours) | Hybrid | 22.044 | 0.7429 | 0.430 |

## 4. Performance across upsampling factors (RMSE, cm)

| Method | x2 | x4 | x8 | x16 |
|---|---|---|---|---|
| Nearest | 2.405 | 4.748 | 8.044 | 12.700 |
| Bilinear | 1.665 | 3.286 | 5.756 | 9.313 |
| Bicubic | 1.539 | 3.230 | 5.820 | 9.555 |
| Joint Bilateral (JBU) | 1.724 | 3.353 | 5.518 | 8.627 |
| Guided Filter | 2.910 | 4.932 | 7.274 | 9.101 |
| Classical pipeline (ours) | 3.623 | 4.721 | 6.444 | 9.074 |
| Deep learning only (TGNet) | 1.248 | 2.503 | 4.363 | 7.023 |
| Hybrid (ours) | 1.805 | 2.785 | 4.469 | 6.881 |

**Noisy + missing depth (robustness)**

x4

| Method | Type | RMSE (cm) | MAE (cm) | BoundaryRMSE (cm) | EdgeF1 |
|---|---|---|---|---|---|
| Nearest | Interpolation | 62.167 | 17.342 | 58.501 | 0.6649 |
| Bilinear | Interpolation | 61.786 | 15.689 | 55.993 | 0.5565 |
| Bicubic | Interpolation | 61.888 | 16.554 | 56.125 | 0.6234 |
| Joint Bilateral (JBU) | Classical refinement | 6.264 | 2.875 | 17.732 | 0.8716 |
| Guided Filter | Classical refinement | 7.530 | 3.060 | 23.839 | 0.5486 |
| Classical pipeline (ours) | Classical refinement | 7.200 | 2.603 | 22.685 | 0.9135 |
| Deep learning only (TGNet) | Deep learning | 22.878 | 5.880 | 24.585 | 0.6949 |
| Hybrid (ours) | Hybrid | 5.485 | 2.071 | 15.921 | 0.9373 |

x8

| Method | Type | RMSE (cm) | MAE (cm) | BoundaryRMSE (cm) | EdgeF1 |
|---|---|---|---|---|---|
| Nearest | Interpolation | 62.081 | 17.820 | 68.945 | 0.5119 |
| Bilinear | Interpolation | 61.113 | 16.018 | 64.076 | 0.3256 |
| Bicubic | Interpolation | 61.445 | 16.948 | 64.404 | 0.3772 |
| Joint Bilateral (JBU) | Classical refinement | 8.436 | 3.903 | 25.053 | 0.7600 |
| Guided Filter | Classical refinement | 10.063 | 4.420 | 30.885 | 0.4284 |
| Classical pipeline (ours) | Classical refinement | 9.073 | 3.655 | 28.440 | 0.7497 |
| Deep learning only (TGNet) | Deep learning | 23.715 | 6.666 | 30.818 | 0.6335 |
| Hybrid (ours) | Hybrid | 7.258 | 2.909 | 21.867 | 0.8397 |


## 5. Processing latency and memory (x4, cuda)

| Method | Latency (ms / image) | Peak extra memory (MB, CPU) |
|---|---|---|
| Nearest | 1.8 | 0.0 |
| Bilinear | 0.9 | 0.0 |
| Bicubic | 1.1 | 0.0 |
| Joint Bilateral (JBU) | 95.8 | 0.0 |
| Guided Filter | 11.5 | 0.0 |
| Classical pipeline (ours) | 130.2 | 0.0 |
| Deep learning only (TGNet) | 126.8 | 0.0 |
| Hybrid (ours) | 257.6 | 0.0 |

Model: TGNet, 1.99 M parameters, ~7.6 MB (fp32).
