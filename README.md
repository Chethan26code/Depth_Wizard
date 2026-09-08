# DepthWizard — Phase 1 (M1) Height Prior

You own **M1**: a frozen Depth Anything V2 backbone that outputs a *relative* height field `ĥ` plus uncertainty. Metres come later in M2.

## Run in this order (learn-by-doing)

```bash
python -m m1.lesson0_scale          # no neural net — the geometry
python -m m1.zeroshot               # DA-V2 Small, first RMSE number
python -m m1.train --epochs 1 --tiles 8   # train decoder on synthetic tiles
```

