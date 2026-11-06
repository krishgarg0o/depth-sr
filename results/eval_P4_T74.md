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
| Classical pipeline (ours) | Classical refinement | 1.496 | 0.430 |
| Deep learning only (TGNet) | Deep learning | 1.137 | 0.364 |
| Hybrid (ours) | Hybrid | 1.185 | 0.367 |

**x4**

| Method | Type | RMSE (cm) | MAE (cm) |
|---|---|---|---|
| Nearest | Interpolation | 4.748 | 1.272 |
| Bilinear | Interpolation | 3.286 | 0.922 |
| Bicubic | Interpolation | 3.230 | 0.976 |
| Joint Bilateral (JBU) | Classical refinement | 3.353 | 1.104 |
| Guided Filter | Classical refinement | 4.932 | 1.869 |
| Classical pipeline (ours) | Classical refinement | 3.057 | 0.939 |
| Deep learning only (TGNet) | Deep learning | 2.475 | 0.747 |
| Hybrid (ours) | Hybrid | 2.442 | 0.740 |

**x8**

| Method | Type | RMSE (cm) | MAE (cm) |
|---|---|---|---|
| Nearest | Interpolation | 8.044 | 2.423 |
| Bilinear | Interpolation | 5.756 | 1.837 |
| Bicubic | Interpolation | 5.820 | 2.006 |
| Joint Bilateral (JBU) | Classical refinement | 5.518 | 2.008 |
| Guided Filter | Classical refinement | 7.274 | 3.027 |
| Classical pipeline (ours) | Classical refinement | 5.528 | 1.942 |
| Deep learning only (TGNet) | Deep learning | 4.378 | 1.466 |
| Hybrid (ours) | Hybrid | 4.328 | 1.431 |

**x16**

| Method | Type | RMSE (cm) | MAE (cm) |
|---|---|---|---|
| Nearest | Interpolation | 12.700 | 4.546 |
| Bilinear | Interpolation | 9.313 | 3.449 |
| Bicubic | Interpolation | 9.555 | 3.779 |
| Joint Bilateral (JBU) | Classical refinement | 8.627 | 3.453 |
| Guided Filter | Classical refinement | 9.101 | 3.815 |
| Classical pipeline (ours) | Classical refinement | 8.814 | 3.532 |
| Deep learning only (TGNet) | Deep learning | 7.045 | 2.590 |
| Hybrid (ours) | Hybrid | 6.940 | 2.554 |

## 2. PSNR and SSIM

**x2**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 46.497 | 0.9940 |
| Bilinear | Interpolation | 49.797 | 0.9972 |
| Bicubic | Interpolation | 50.516 | 0.9975 |
| Joint Bilateral (JBU) | Classical refinement | 49.501 | 0.9967 |
| Guided Filter | Classical refinement | 44.898 | 0.9918 |
| Classical pipeline (ours) | Classical refinement | 50.753 | 0.9976 |
| Deep learning only (TGNet) | Deep learning | 53.223 | 0.9985 |
| Hybrid (ours) | Hybrid | 52.825 | 0.9984 |

**x4**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 40.579 | 0.9791 |
| Bilinear | Interpolation | 43.865 | 0.9909 |
| Bicubic | Interpolation | 44.025 | 0.9905 |
| Joint Bilateral (JBU) | Classical refinement | 43.710 | 0.9898 |
| Guided Filter | Classical refinement | 40.314 | 0.9834 |
| Classical pipeline (ours) | Classical refinement | 44.501 | 0.9912 |
| Deep learning only (TGNet) | Deep learning | 46.402 | 0.9945 |
| Hybrid (ours) | Hybrid | 46.483 | 0.9946 |

**x8**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 35.993 | 0.9554 |
| Bilinear | Interpolation | 38.998 | 0.9805 |
| Bicubic | Interpolation | 38.892 | 0.9787 |
| Joint Bilateral (JBU) | Classical refinement | 39.394 | 0.9797 |
| Guided Filter | Classical refinement | 36.939 | 0.9761 |
| Classical pipeline (ours) | Classical refinement | 39.344 | 0.9796 |
| Deep learning only (TGNet) | Deep learning | 41.437 | 0.9874 |
| Hybrid (ours) | Hybrid | 41.558 | 0.9877 |

**x16**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 31.989 | 0.9327 |
| Bilinear | Interpolation | 34.793 | 0.9708 |
| Bicubic | Interpolation | 34.592 | 0.9685 |
| Joint Bilateral (JBU) | Classical refinement | 35.495 | 0.9712 |
| Guided Filter | Classical refinement | 35.007 | 0.9731 |
| Classical pipeline (ours) | Classical refinement | 35.263 | 0.9720 |
| Deep learning only (TGNet) | Deep learning | 37.352 | 0.9798 |
| Hybrid (ours) | Hybrid | 37.475 | 0.9800 |

## 3. Edge preservation and boundary accuracy

**x2**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 8.292 | 0.9761 | 0.253 |
| Bilinear | Interpolation | 5.785 | 0.9233 | 0.255 |
| Bicubic | Interpolation | 5.338 | 0.9749 | 0.266 |
| Joint Bilateral (JBU) | Classical refinement | 5.864 | 0.9515 | 0.276 |
| Guided Filter | Classical refinement | 9.906 | 0.8567 | 0.360 |
| Classical pipeline (ours) | Classical refinement | 5.179 | 0.9741 | 0.263 |
| Deep learning only (TGNet) | Deep learning | 3.765 | 0.9904 | 0.237 |
| Hybrid (ours) | Hybrid | 3.971 | 0.9918 | 0.237 |

**x4**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 16.376 | 0.9598 | 0.310 |
| Bilinear | Interpolation | 11.488 | 0.7578 | 0.303 |
| Bicubic | Interpolation | 11.276 | 0.8238 | 0.351 |
| Joint Bilateral (JBU) | Classical refinement | 11.409 | 0.8823 | 0.392 |
| Guided Filter | Classical refinement | 16.525 | 0.7028 | 0.502 |
| Classical pipeline (ours) | Classical refinement | 10.615 | 0.8613 | 0.348 |
| Deep learning only (TGNet) | Deep learning | 8.349 | 0.9571 | 0.262 |
| Hybrid (ours) | Hybrid | 8.268 | 0.9701 | 0.264 |

**x8**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 27.518 | 0.7364 | 0.386 |
| Bilinear | Interpolation | 19.989 | 0.4246 | 0.357 |
| Bicubic | Interpolation | 19.812 | 0.4822 | 0.424 |
| Joint Bilateral (JBU) | Classical refinement | 18.443 | 0.7611 | 0.564 |
| Guided Filter | Classical refinement | 23.204 | 0.5419 | 0.618 |
| Classical pipeline (ours) | Classical refinement | 18.709 | 0.6285 | 0.438 |
| Deep learning only (TGNet) | Deep learning | 14.556 | 0.8741 | 0.330 |
| Hybrid (ours) | Hybrid | 14.433 | 0.8977 | 0.336 |

**x16**

| Method | Type | BoundaryRMSE (cm) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 41.942 | 0.3632 | 0.410 |
| Bilinear | Interpolation | 30.456 | 0.2043 | 0.353 |
| Bicubic | Interpolation | 30.592 | 0.2601 | 0.420 |
| Joint Bilateral (JBU) | Classical refinement | 27.292 | 0.5607 | 0.636 |
| Guided Filter | Classical refinement | 28.328 | 0.3767 | 0.551 |
| Classical pipeline (ours) | Classical refinement | 28.041 | 0.4318 | 0.487 |
| Deep learning only (TGNet) | Deep learning | 22.705 | 0.7200 | 0.428 |
| Hybrid (ours) | Hybrid | 22.263 | 0.7582 | 0.458 |

## 4. Performance across upsampling factors (RMSE, cm)

| Method | x2 | x4 | x8 | x16 |
|---|---|---|---|---|
| Nearest | 2.405 | 4.748 | 8.044 | 12.700 |
| Bilinear | 1.665 | 3.286 | 5.756 | 9.313 |
| Bicubic | 1.539 | 3.230 | 5.820 | 9.555 |
| Joint Bilateral (JBU) | 1.724 | 3.353 | 5.518 | 8.627 |
| Guided Filter | 2.910 | 4.932 | 7.274 | 9.101 |
| Classical pipeline (ours) | 1.496 | 3.057 | 5.528 | 8.814 |
| Deep learning only (TGNet) | 1.137 | 2.475 | 4.378 | 7.045 |
| Hybrid (ours) | 1.185 | 2.442 | 4.328 | 6.940 |

**Noisy + missing depth (robustness)**

x4

| Method | Type | RMSE (cm) | MAE (cm) | BoundaryRMSE (cm) | EdgeF1 |
|---|---|---|---|---|---|
| Nearest | Interpolation | 62.167 | 17.342 | 58.501 | 0.6649 |
| Bilinear | Interpolation | 61.786 | 15.689 | 55.993 | 0.5565 |
| Bicubic | Interpolation | 61.888 | 16.554 | 56.125 | 0.6234 |
| Joint Bilateral (JBU) | Classical refinement | 6.264 | 2.875 | 17.732 | 0.8716 |
| Guided Filter | Classical refinement | 7.530 | 3.060 | 23.839 | 0.5486 |
| Classical pipeline (ours) | Classical refinement | 5.997 | 2.484 | 17.748 | 0.8537 |
| Deep learning only (TGNet) | Deep learning | 22.875 | 5.908 | 24.508 | 0.6973 |
| Hybrid (ours) | Hybrid | 5.456 | 2.086 | 15.880 | 0.9308 |

x8

| Method | Type | RMSE (cm) | MAE (cm) | BoundaryRMSE (cm) | EdgeF1 |
|---|---|---|---|---|---|
| Nearest | Interpolation | 62.081 | 17.820 | 68.945 | 0.5119 |
| Bilinear | Interpolation | 61.113 | 16.018 | 64.076 | 0.3256 |
| Bicubic | Interpolation | 61.445 | 16.948 | 64.404 | 0.3772 |
| Joint Bilateral (JBU) | Classical refinement | 8.436 | 3.903 | 25.053 | 0.7600 |
| Guided Filter | Classical refinement | 10.063 | 4.420 | 30.885 | 0.4284 |
| Classical pipeline (ours) | Classical refinement | 8.281 | 3.605 | 25.267 | 0.6800 |
| Deep learning only (TGNet) | Deep learning | 23.734 | 6.759 | 30.884 | 0.6350 |
| Hybrid (ours) | Hybrid | 7.370 | 2.933 | 22.359 | 0.8337 |


## 5. Processing latency and memory (x4, cuda)

| Method | Latency (ms / image) | Peak extra memory (MB, CPU) |
|---|---|---|
| Nearest | 1.8 | 0.0 |
| Bilinear | 0.8 | 0.0 |
| Bicubic | 1.1 | 0.0 |
| Joint Bilateral (JBU) | 96.3 | 0.0 |
| Guided Filter | 11.3 | 0.0 |
| Classical pipeline (ours) | 128.3 | 0.0 |
| Deep learning only (TGNet) | 123.2 | 0.0 |
| Hybrid (ours) | 252.5 | 0.0 |

Model: TGNet, 1.99 M parameters, ~7.6 MB (fp32).
