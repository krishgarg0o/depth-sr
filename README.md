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

## Results
**Team T74:** Jidvish Gupta (1RUA24CSE0186), Keerthana Kiran (1RUA24CSE0197), Krish Garg (1RUA24CSE0208), Krishaa Daga (1RUA24CSE0209)
· **Report:** T74_P4_Report.pdf (submitted separately)

RMSE in cm on 100 unseen NYU Depth V2 test images (indices 1000-1099, never used for training or model selection).
Full tables incl. MAE, PSNR, SSIM, boundary RMSE, edge F1, texture copy, latency, memory: [results/eval_P4_T74.md](results/eval_P4_T74.md).

| Method | Type | x2 | x4 | x8 | x16 | x4 noisy+holes | x8 noisy+holes |
|---|---|---|---|---|---|---|---|
| Nearest | Interpolation | 2.41 | 4.75 | 8.04 | 12.70 | 62.17 | 62.08 |
| Bilinear | Interpolation | 1.67 | 3.29 | 5.76 | 9.31 | 61.79 | 61.11 |
| Bicubic | Interpolation | 1.54 | 3.23 | 5.82 | 9.56 | 61.89 | 61.45 |
| Joint bilateral (JBU) | Classical | 1.72 | 3.35 | 5.52 | 8.63 | 6.26 | 8.44 |
| Guided filter | Classical | 2.91 | 4.93 | 7.27 | 9.10 | 7.53 | 10.06 |
| Classical pipeline (ours, adaptive v2) | Classical | 1.50 | 3.06 | 5.53 | 8.81 | 6.00 | 8.28 |
| TGNet only | Deep learning | **1.14** | 2.48 | 4.38 | 7.05 | 22.88 | 23.73 |
| **Hybrid (ours)** | Hybrid | 1.19 | **2.44** | **4.33** | **6.94** | **5.46** | **7.37** |

**Generalisation to an unseen dataset** - 26 Middlebury 2005/2006 stereo scenes (different sensor and domain, disparity units),
never used for training ([results/eval_middlebury.md](results/eval_middlebury.md)):

| Method | x2 | x4 | x8 | x4 noisy+holes | x8 noisy+holes |
|---|---|---|---|---|---|
| Bicubic | 2.13 | 3.64 | 5.52 | 26.52 | 26.61 |
| Joint bilateral | 1.98 | 2.95 | 4.24 | 3.78 | 4.81 |
| Classical pipeline (ours) | 1.97 | 3.12 | 4.76 | 3.71 | 4.80 |
| TGNet only | 1.64 | 3.06 | 4.41 | 10.79 | 11.01 |
| **Hybrid (ours)** | **1.59** | **2.69** | **4.02** | **3.52** | **4.50** |

* **The hybrid is the most accurate method from x4 to x16** on NYU, with the best boundary RMSE, PSNR, SSIM and edge F1 at x8
  (edge F1 0.898 vs 0.482 for bicubic); TGNet alone is marginally better only at x2 (1.14 vs 1.19 cm).
* **Robust to noise and holes**: 9-11 % lower RMSE than the best classical method and 3-4x lower than TGNet alone.
* **Generalises**: best method in every Middlebury setting.
* **Ablation** ([results/ablation_nyu.md](results/ablation_nyu.md), 30 images): removing the trust map raises RMSE by 5-28 % and
  texture copying by 18-54 %; removing noise-adaptive smoothing +19 % (clean x4); re-enabling the v1 boundary snapping +8-57 %;
  removing hole filling breaks noisy input (5.5 -> 54 cm).
* **Classical stage v2**: our first classical stage (boundary snapping, always-on smoothing) lost to bicubic on clean input
  (x2: 3.62 vs 1.54 cm). The adaptive v2 stage beats bicubic at every factor, and retraining on it improved the hybrid
  (x2 1.81 -> 1.19, x4 2.79 -> 2.44, x8 4.47 -> 4.33 cm; x16 6.88 -> 6.94 and noisy x8 7.26 -> 7.37 are slightly worse).
* Training: validation RMSE of the network is 52.0 % lower than its classical input (results/train_log.csv).

## Quick start
```bash
pip install -r requirements.txt
python scripts/run_inference.py --input_dir ./samples/ --output_dir ./results/inference/ --scale 8   # batch CLI
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
python scripts/evaluate.py --dataset middlebury --data <middlebury folder> --scales 2,4,8 --limit 30   # unseen domain (Kaggle: minhanhtruong/middleburystereodataset)
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
Trained on indoor Kinect data only; outdoor LiDAR is out of distribution. The Middlebury evaluation uses a Kaggle mirror of the 2005/2006 datasets (the original server refuses automated downloads). Thin structures narrower than one low-resolution pixel cannot be recovered at x8/x16 (see results/failure_cases_nyu.md). Synthetic degradation is simpler than
real sensor noise. Same-colour foreground/background boundaries cannot be recovered from RGB. See
[design doc §12-13](docs/design.md) for limitations and ethics.

## Datasets
NYU Depth V2 (Silberman et al., ECCV 2012), Middlebury 2005 stereo (Scharstein & Pal, CVPR 2007).
