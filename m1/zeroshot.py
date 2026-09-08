"""Zero-shot Depth Anything V2 → affine-calibrated metres → RMSE.

This is the Week-1 ML-lead experiment from the blueprint:

    "Off-the-shelf gets X m; here is why our four ideas move it."

On this laptop we run ViT-Small on synthetic tiles (and any RGB you pass).
Swap `--size large` on a GPU box / Colab for the submission backbone.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from m1.affine import fit_affine
from m1.config import M1Config
from m1.metrics import delta_accuracy, mae, rmse
from m1.model import HeightPriorNetwork, pick_device
from m1.synthetic import make_city_tile, tile_to_pil


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 zero-shot baseline")
    p.add_argument("--size", default="small", choices=["small", "base", "large"])
    p.add_argument("--image", type=Path, default=None, help="Optional RGB path")
    p.add_argument("--height", type=Path, default=None, help="Optional .npy height map (metres)")
    p.add_argument("--gsd", type=float, default=1.0)
    p.add_argument("--tiles", type=int, default=4, help="Synthetic tiles if no --image")
    p.add_argument("--out", type=Path, default=Path("outputs/zeroshot"))
    return p.parse_args()


def relative_from_official_head(net: HeightPriorNetwork, image: Image.Image) -> np.ndarray:
    """Use the *stock* DA-V2 head, no FiLM — true zero-shot.

    First run should measure the frozen model as published, before we claim
    credit for our decoder / GSD conditioning.
    """
    device = next(net.parameters()).device
    pv = net.preprocess(image, device)
    with torch.no_grad():
        out = net.da(pixel_values=pv)
        pred = out.predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return pred.detach().cpu().numpy().astype(np.float32)


def save_preview(rgb: np.ndarray, relative: np.ndarray, metric: np.ndarray | None, path: Path) -> None:
    cols = 3 if metric is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[1].imshow(relative, cmap="magma")
    axes[1].set_title("Relative ĥ (unitless)")
    if metric is not None:
        im = axes[2].imshow(metric, cmap="terrain")
        axes[2].set_title("Affine-scaled (m)")
        fig.colorbar(im, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device={device}  backbone={args.size}")
    net = HeightPriorNetwork(M1Config(size=args.size, freeze_backbone=True)).to(device)
    net.eval()
    trainable, total = net.trainable_parameter_count()
    print(f"params: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

    rmses, maes = [], []

    if args.image is not None:
        image = Image.open(args.image).convert("RGB")
        relative = relative_from_official_head(net, image)
        gt = np.load(args.height) if args.height else None
        metric = None
        if gt is not None:
            fit = fit_affine(relative, gt)
            metric = fit.apply(relative)
            print(
                f"affine a={fit.a:.4f}  b={fit.b:.4f}  "
                f"RMSE={rmse(metric, gt):.3f} m  MAE={mae(metric, gt):.3f} m"
            )
        save_preview(np.array(image), relative, metric, args.out / "preview.png")
        np.save(args.out / "relative.npy", relative)
        return

    print("no --image given; running synthetic city tiles (learning stand-in for GAMUS)")
    for i in range(args.tiles):
        tile = make_city_tile(gsd_m=args.gsd, seed=1000 + i)
        image = tile_to_pil(tile)
        relative = relative_from_official_head(net, image)
        fit = fit_affine(relative, tile.height_m)
        metric = fit.apply(relative)
        r, m = rmse(metric, tile.height_m), mae(metric, tile.height_m)
        d = delta_accuracy(metric, tile.height_m)
        rmses.append(r)
        maes.append(m)
        print(
            f"tile {i:02d}  buildings={tile.building_count:2d}  "
            f"a={fit.a:+.4f}  b={fit.b:+.2f}  RMSE={r:.3f} m  MAE={m:.3f} m  d<1.25={d:.3f}"
        )
        save_preview(tile.rgb, relative, metric, args.out / f"tile_{i:02d}.png")

    print("-" * 60)
    print(f"ZERO-SHOT BASELINE  mean RMSE={np.mean(rmses):.3f} m  mean MAE={np.mean(maes):.3f} m")
    print("This number is the floor. Fine-tuning + GSD FiLM + M2 fusion should beat it.")
    (args.out / "baseline.txt").write_text(
        f"mean_rmse_m={float(np.mean(rmses)):.6f}\nmean_mae_m={float(np.mean(maes)):.6f}\n"
        f"backbone={args.size}\ndevice={device}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
