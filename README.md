# DepthSR - Edge-Preserving Depth Map Upsampling and Refinement (P4)

**Live demo:** https://depthsr-t74.streamlit.app · **Design doc:** [docs/design.md](docs/design.md) ·
**Results:** [results/eval_P4_T74.md](results/eval_P4_T74.md) · **Ablation:** [results/ablation_nyu.md](results/ablation_nyu.md) ·
**Failure cases:** [results/failure_cases_nyu.md](results/failure_cases_nyu.md)

Turns low-resolution, sparse or noisy depth maps into dense high-resolution depth (x2, x4, x8, x16), using the RGB
image as a guide **only where it marks a real depth boundary** - sharp object edges, no texture copying.

![architecture](docs/architecture.png)

**Pipeline:** hole filling (normalised convolution) -> bicubic -> RGB edges + depth-discontinuity band ->
**trust map** -> trust-gated joint bilateral / guided filtering -> boundary snapping -> **TGNet** (1.99 M param
two-branch guided U-Net) predicts a residual and a per-pixel fusion gate: `D = D_classical + alpha * r`.

## Results (NYU Depth V2, 449 unseen test images, RMSE in cm)
100 unseen test images (indices 1000-1099, never used for training or model selection). Full tables incl. MAE, PSNR, SSIM,
boundary RMSE, edge F1, texture copy, latency: [results/eval_P4_T74.md](results/eval_P4_T74.md).

| Method | Type | x2 | x4 | x8 | x16 | x4 noisy+holes | x8 noisy+holes |
|---|---|---|---|---|---|---|---|
| Nearest | Interpolation | 2.41 | 4.75 | 8.04 | 12.70 | 62.17 | 62.08 |
| Bilinear | Interpolation | 1.67 | 3.29 | 5.76 | 9.31 | 61.79 | 61.11 |
| Bicubic | Interpolation | 1.54 | 3.23 | 5.82 | 9.56 | 61.89 | 61.45 |
| Joint bilateral (JBU) | Classical | 1.72 | 3.35 | 5.52 | 8.63 | 6.26 | 8.44 |
| Guided filter | Classical | 2.91 | 4.93 | 7.27 | 9.10 | 7.53 | 10.06 |
| Classical pipeline (ours) | Classical | 3.62 | 4.72 | 6.44 | 9.07 | 7.20 | 9.07 |
| TGNet only | Deep learning | **1.25** | **2.50** | **4.36** | 7.02 | 22.88 | 23.72 |
| **Hybrid (ours)** | Hybrid | 1.81 | 2.79 | 4.47 | **6.88** | **5.49** | **7.26** |

* **Hybrid is the most robust method**: best at x16 and best on noisy depth with missing values
  (12-14 % lower RMSE than the best classical filter, 3-4x lower than TGNet alone, which gets no hole-filled input).
* TGNet alone is best on clean input at x2-x8; the hybrid is 0.11 cm behind it at x8 and has the best edge F1 at x8/x16
  (0.889 / 0.743 vs 0.482 / 0.260 for bicubic).
* Ablation ([results/ablation_nyu.md](results/ablation_nyu.md), 30 images): removing the **trust map** raises hybrid RMSE by
  9-28 % and texture copying by 18-48 %; removing **hole filling** breaks the noisy case (RMSE 5.5 -> 55 cm).
* Training: validation RMSE of the network is **52.5 % lower than its classical input** (results/train_log.csv).

## Quick start
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py   # deployed version (Streamlit Community Cloud)
python app.py                      # Gradio version, http://localhost:7860
# or
docker build -t depthsr . && docker run -p 7860:7860 depthsr
python tests/test_pipeline.py      # smoke tests
```

## Reproduce training and evaluation
On a GPU machine or the free Kaggle T4 (`notebooks/kaggle_train_eval.ipynb` does all of this):
```bash
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
python scripts/prepare_nyu.py --mat nyu_depth_v2_labeled.mat --out data     # split 900/100/449
python scripts/train.py --data data --iters 6000 --bs 8                     # -> checkpoints/tgnet.pt (fp32, ~60 min on a T4)
python scripts/evaluate.py --dataset nyu --data data --limit 100 --noisy_limit 30 --team T74 --usns "1RUA24CSE0186,1RUA24CSE0197,1RUA24CSE0208,1RUA24CSE0209"
python scripts/evaluate.py --dataset middlebury --data middlebury --scales 2,4,8   # unseen domain
```
Seeds are fixed (training seed 0, degradation seed = image index), so the numbers are reproducible.

## Repository layout
```
app.py                 Gradio app (validation, pipeline stages, comparison, 3D view, downloads)
depthsr/io_utils.py    loading depth (png16/KITTI/8-bit/npy/pfm/tiff), validation, colouring
depthsr/degrade.py     degradation protocol (downsample, noise, holes, sparsity)
depthsr/classical.py   baselines, hole filling, adaptive params, JBU, guided filter, edges, trust map, snapping
depthsr/model.py       TGNet
depthsr/hybrid.py      DepthSR: runs any of the 8 methods
depthsr/metrics.py     RMSE, MAE, PSNR, SSIM, boundary RMSE, edge F1, texture copy
depthsr/data.py        training dataset with on-the-fly degradation
scripts/               prepare_nyu.py, train.py, evaluate.py
docs/                  design.md, architecture.mmd/.png
results/               eval_P4_T74.md, ablation, failure cases, figures, train log
```

## Methods compared
| Family | Methods |
|---|---|
| Interpolation | nearest, bilinear, bicubic |
| Classical refinement | joint bilateral upsampling, guided filter, our classical pipeline |
| Deep learning | TGNet without classical stage (bicubic input) |
| Hybrid | classical stage + TGNet + fusion gate |

## Limitations
Trained on indoor Kinect data only; outdoor LiDAR is out of distribution. Our classical pipeline alone is worse than bicubic on *clean* input (its boundary snapping and smoothing pay off only on noisy / incomplete depth). The Middlebury cross-dataset test could not be run (the Middlebury server returned HTTP 403 to the Kaggle machine); generalisation is only measured on unseen NYU test images. Synthetic degradation is simpler than
real sensor noise. Same-colour foreground/background boundaries cannot be recovered from RGB. See
[design doc §12-13](docs/design.md) for limitations and ethics.

## Datasets
NYU Depth V2 (Silberman et al., ECCV 2012), Middlebury 2005 stereo (Scharstein & Pal, CVPR 2007).
