# Team plan (deadline: tomorrow 2 PM)

## Order of work
| When | Who | Task |
|---|---|---|
| Tonight, first 30 min | M4 | Create GitHub repo, push this folder. Create Hugging Face Space (Docker SDK), push the same code -> live URL works with the classical methods even before training. |
| Tonight, right after | M2 | Kaggle: new notebook -> import `notebooks/kaggle_train_eval.ipynb`, GPU T4 + Internet ON, set REPO_URL / TEAM / USNS, Run All (~2 h, unattended). |
| While Kaggle runs | M1 | Read `depthsr/classical.py`; run the app locally and screenshot each pipeline stage; be able to explain every map. |
| While Kaggle runs | M3 | Read `metrics.py` + `evaluate.py`; run `python scripts/evaluate.py --dataset synthetic --limit 8` locally to see the outputs. |
| After Kaggle | M2 + M3 | Download `depthsr_outputs.zip`, unzip into the repo (checkpoints/, results/, samples/), commit + push to GitHub **and** the HF Space. |
| Morning | M3 + M4 | Paste result tables into README + design doc §8-9, write 2-3 lines per failure case after looking at the figures, fill names/USNs/URLs. |
| Morning | all | Rehearse the 3-minute demo twice on a phone and a laptop. Submit by 1:30 PM. |

## 3-minute demo script
1. (20 s) Problem: LR depth blurs edges and creates flying pixels; naive RGB guidance copies texture.
2. (40 s) Pick an NYU example, x8 -> show slider, then "Pipeline stages": trust map vs texture edges.
3. (40 s) "Method comparison": table, hybrid best RMSE + boundary RMSE; latency.
4. (30 s) "3D point cloud": bicubic flying pixels vs ours.
5. (30 s) Judge's input: upload anything (RGB + depth, or RGB only), add holes/noise with the sliders.
6. (20 s) One failure case + why.

## Likely viva questions
* Why log-depth gradients for discontinuities? (relative jump, works for near and far objects)
* Why does JBU copy texture and how does the trust map stop it?
* Why predict a residual instead of depth? What does alpha mean?
* How is one model handling x2-x16? (log2-scale input plane, mixed-scale training)
* How do you get the DL-only baseline? (same net trained 25 % of the time without the classical inputs)
* What is boundary RMSE / edge F1 / texture copy, and why not only RMSE?
* What is your train/val/test split, and is the test set ever used for model selection? (No - val only)
* Latency/memory on CPU and how you would make it faster (ONNX, fp16, smaller radius).
