USN1,USN2,USN3,USN4

# Evaluation - P4 Edge-Preserving Depth Map Upsampling (T01)

Dataset: **middlebury** (26 unseen test images), device: cuda. Degradation: bicubic downsampling (clean); + 2 % depth-relative Gaussian noise and 4 % random holes (noisy).
PSNR/SSIM on depth normalised by GT max. BoundaryRMSE = RMSE within 5 px of GT depth discontinuities. EdgeF1 = F1 of depth-edge maps (2 px tolerance). TextureCopy = spurious depth gradient on flat-but-textured regions (lower is better).

## 1. Depth accuracy - RMSE and MAE

**x2**

| Method | Type | RMSE (disparity px) | MAE (disparity px) |
|---|---|---|---|
| Nearest | Interpolation | 2.583 | 0.530 |
| Bilinear | Interpolation | 2.201 | 0.533 |
| Bicubic | Interpolation | 2.128 | 0.574 |
| Joint Bilateral (JBU) | Classical refinement | 1.978 | 0.512 |
| Guided Filter | Classical refinement | 2.840 | 0.870 |
| Classical pipeline (ours) | Classical refinement | 1.968 | 0.509 |
| Deep learning only (TGNet) | Deep learning | 1.644 | 0.417 |
| Hybrid (ours) | Hybrid | 1.592 | 0.401 |

**x4**

| Method | Type | RMSE (disparity px) | MAE (disparity px) |
|---|---|---|---|
| Nearest | Interpolation | 4.512 | 0.921 |
| Bilinear | Interpolation | 3.569 | 0.940 |
| Bicubic | Interpolation | 3.642 | 1.075 |
| Joint Bilateral (JBU) | Classical refinement | 2.949 | 0.880 |
| Guided Filter | Classical refinement | 4.159 | 1.578 |
| Classical pipeline (ours) | Classical refinement | 3.116 | 0.937 |
| Deep learning only (TGNet) | Deep learning | 3.056 | 0.756 |
| Hybrid (ours) | Hybrid | 2.689 | 0.710 |

**x8**

| Method | Type | RMSE (disparity px) | MAE (disparity px) |
|---|---|---|---|
| Nearest | Interpolation | 6.868 | 1.677 |
| Bilinear | Interpolation | 5.332 | 1.684 |
| Bicubic | Interpolation | 5.518 | 1.939 |
| Joint Bilateral (JBU) | Classical refinement | 4.243 | 1.537 |
| Guided Filter | Classical refinement | 5.804 | 2.640 |
| Classical pipeline (ours) | Classical refinement | 4.756 | 1.706 |
| Deep learning only (TGNet) | Deep learning | 4.412 | 1.254 |
| Hybrid (ours) | Hybrid | 4.020 | 1.208 |

## 2. PSNR and SSIM

**x2**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 38.425 | 0.9531 |
| Bilinear | Interpolation | 39.814 | 0.9569 |
| Bicubic | Interpolation | 40.095 | 0.9573 |
| Joint Bilateral (JBU) | Classical refinement | 40.666 | 0.9592 |
| Guided Filter | Classical refinement | 37.567 | 0.9473 |
| Classical pipeline (ours) | Classical refinement | 40.748 | 0.9595 |
| Deep learning only (TGNet) | Deep learning | 42.374 | 0.9627 |
| Hybrid (ours) | Hybrid | 42.599 | 0.9632 |

**x4**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 33.539 | 0.9259 |
| Bilinear | Interpolation | 35.597 | 0.9399 |
| Bicubic | Interpolation | 35.429 | 0.9364 |
| Joint Bilateral (JBU) | Classical refinement | 37.186 | 0.9472 |
| Guided Filter | Classical refinement | 34.253 | 0.9303 |
| Classical pipeline (ours) | Classical refinement | 36.748 | 0.9433 |
| Deep learning only (TGNet) | Deep learning | 36.988 | 0.9489 |
| Hybrid (ours) | Hybrid | 38.017 | 0.9522 |

**x8**

| Method | Type | PSNR | SSIM |
|---|---|---|---|
| Nearest | Interpolation | 29.803 | 0.8919 |
| Bilinear | Interpolation | 32.095 | 0.9177 |
| Bicubic | Interpolation | 31.803 | 0.9103 |
| Joint Bilateral (JBU) | Classical refinement | 34.075 | 0.9306 |
| Guided Filter | Classical refinement | 31.349 | 0.9151 |
| Classical pipeline (ours) | Classical refinement | 33.055 | 0.9210 |
| Deep learning only (TGNet) | Deep learning | 33.800 | 0.9352 |
| Hybrid (ours) | Hybrid | 34.564 | 0.9381 |

## 3. Edge preservation and boundary accuracy

**x2**

| Method | Type | BoundaryRMSE (disparity px) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 6.244 | 0.8875 | 0.718 |
| Bilinear | Interpolation | 5.340 | 0.8521 | 0.723 |
| Bicubic | Interpolation | 5.169 | 0.8845 | 0.848 |
| Joint Bilateral (JBU) | Classical refinement | 4.786 | 0.8867 | 0.647 |
| Guided Filter | Classical refinement | 6.814 | 0.8180 | 0.905 |
| Classical pipeline (ours) | Classical refinement | 4.771 | 0.8886 | 0.683 |
| Deep learning only (TGNet) | Deep learning | 3.927 | 0.8921 | 0.492 |
| Hybrid (ours) | Hybrid | 3.808 | 0.8998 | 0.445 |

**x4**

| Method | Type | BoundaryRMSE (disparity px) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 10.901 | 0.8858 | 0.962 |
| Bilinear | Interpolation | 8.679 | 0.7284 | 0.884 |
| Bicubic | Interpolation | 8.864 | 0.7507 | 1.329 |
| Joint Bilateral (JBU) | Classical refinement | 7.103 | 0.8409 | 0.936 |
| Guided Filter | Classical refinement | 9.801 | 0.6899 | 1.186 |
| Classical pipeline (ours) | Classical refinement | 7.529 | 0.7978 | 0.964 |
| Deep learning only (TGNet) | Deep learning | 7.397 | 0.8498 | 0.745 |
| Hybrid (ours) | Hybrid | 6.441 | 0.8660 | 0.693 |

**x8**

| Method | Type | BoundaryRMSE (disparity px) | EdgeF1 | TextureCopy |
|---|---|---|---|---|
| Nearest | Interpolation | 16.724 | 0.6928 | 0.957 |
| Bilinear | Interpolation | 12.996 | 0.4633 | 0.825 |
| Bicubic | Interpolation | 13.258 | 0.5048 | 1.285 |
| Joint Bilateral (JBU) | Classical refinement | 10.028 | 0.7534 | 1.217 |
| Guided Filter | Classical refinement | 12.978 | 0.5584 | 1.248 |
| Classical pipeline (ours) | Classical refinement | 11.380 | 0.6189 | 1.069 |
| Deep learning only (TGNet) | Deep learning | 10.606 | 0.7577 | 0.842 |
| Hybrid (ours) | Hybrid | 9.585 | 0.7884 | 0.864 |

## 4. Performance across upsampling factors (RMSE, disparity px)

| Method | x2 | x4 | x8 |
|---|---|---|---|
| Nearest | 2.583 | 4.512 | 6.868 |
| Bilinear | 2.201 | 3.569 | 5.332 |
| Bicubic | 2.128 | 3.642 | 5.518 |
| Joint Bilateral (JBU) | 1.978 | 2.949 | 4.243 |
| Guided Filter | 2.840 | 4.159 | 5.804 |
| Classical pipeline (ours) | 1.968 | 3.116 | 4.756 |
| Deep learning only (TGNet) | 1.644 | 3.056 | 4.412 |
| Hybrid (ours) | 1.592 | 2.689 | 4.020 |

**Noisy + missing depth (robustness)**

x4

| Method | Type | RMSE (disparity px) | MAE (disparity px) | BoundaryRMSE (disparity px) | EdgeF1 |
|---|---|---|---|---|---|
| Nearest | Interpolation | 26.734 | 7.613 | 28.587 | 0.7144 |
| Bilinear | Interpolation | 26.457 | 7.031 | 27.665 | 0.6267 |
| Bicubic | Interpolation | 26.518 | 7.444 | 27.780 | 0.6619 |
| Joint Bilateral (JBU) | Classical refinement | 3.784 | 1.561 | 8.511 | 0.8107 |
| Guided Filter | Classical refinement | 5.108 | 2.177 | 11.915 | 0.5475 |
| Classical pipeline (ours) | Classical refinement | 3.710 | 1.401 | 8.515 | 0.7975 |
| Deep learning only (TGNet) | Deep learning | 10.791 | 2.948 | 12.295 | 0.7122 |
| Hybrid (ours) | Hybrid | 3.524 | 1.197 | 8.088 | 0.8228 |

x8

| Method | Type | RMSE (disparity px) | MAE (disparity px) | BoundaryRMSE (disparity px) | EdgeF1 |
|---|---|---|---|---|---|
| Nearest | Interpolation | 27.022 | 8.109 | 30.676 | 0.5582 |
| Bilinear | Interpolation | 26.304 | 7.489 | 28.415 | 0.4177 |
| Bicubic | Interpolation | 26.611 | 8.035 | 28.781 | 0.4546 |
| Joint Bilateral (JBU) | Classical refinement | 4.809 | 2.202 | 10.835 | 0.7331 |
| Guided Filter | Classical refinement | 6.540 | 3.212 | 14.590 | 0.4373 |
| Classical pipeline (ours) | Classical refinement | 4.802 | 2.115 | 10.948 | 0.6763 |
| Deep learning only (TGNet) | Deep learning | 11.005 | 3.496 | 14.171 | 0.6330 |
| Hybrid (ours) | Hybrid | 4.496 | 1.765 | 10.333 | 0.7455 |


## 5. Processing latency and memory (x4, cuda)

| Method | Latency (ms / image) | Peak extra memory (MB, CPU) |
|---|---|---|
| Nearest | 1.2 | nan |
| Bilinear | 0.6 | nan |
| Bicubic | 0.8 | nan |
| Joint Bilateral (JBU) | 52.0 | nan |
| Guided Filter | 6.7 | nan |
| Classical pipeline (ours) | 72.8 | nan |
| Deep learning only (TGNet) | 80.9 | nan |
| Hybrid (ours) | 153.3 | nan |

Model: TGNet, 1.99 M parameters, ~7.6 MB (fp32).
