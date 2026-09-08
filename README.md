# DepthWizard — Phase 1 (M1) Height Prior

You own **M1**: a frozen Depth Anything V2 backbone that outputs a *relative* height field `ĥ` plus uncertainty. Metres come later in M2.

## Run in this order (learn-by-doing)

```bash
python -m m1.lesson0_scale          # no neural net — the geometry
python -m m1.zeroshot               # DA-V2 Small, first RMSE number
python -m m1.train --epochs 1 --tiles 8   # train decoder on synthetic tiles
```

First `zeroshot` / `train` downloads weights from Hugging Face (~100 MB for Small).

## What you should believe after lesson 0

From a satellite, depth is an affine function of height: `H = a·ĥ + b`. Stretching ĥ to 0–1 is the wrong calibration. Least squares for `(a, b)` is the right first calibration.

## Hardware on this machine

No NVIDIA GPU detected. We default to **ViT-Small** on CPU. The SIH blueprint's ViT-L is the same code path: `python -m m1.zeroshot --size large` on Colab/Kaggle with a T4.
