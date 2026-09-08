"""M1 Inference script: Satellite RGB -> Depth Anything V2 + Height Head -> Relative Height Map.

Visualizes:
    RGB satellite image | ground-truth height (if available) | predicted relative height
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from m1.config import M1Config
from m1.model import HeightPriorNetwork, pick_device
from m1.synthetic import make_city_tile, tile_to_pil


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 Height Prior Inference")
    p.add_argument("--size", default="small", choices=["small", "base", "large"])
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/m1_small_best.pt"), help="Path to checkpoint .pt")
    p.add_argument("--image", type=Path, default=None, help="Path to RGB satellite image")
    p.add_argument("--height", type=Path, default=None, help="Optional path to .npy ground truth height (nDSM in m)")
    p.add_argument("--gsd", type=float, default=1.0, help="Ground Sampling Distance in meters")
    p.add_argument("--out", type=Path, default=Path("outputs/visualizations/inference_result.png"), help="Output visualization path")
    return p.parse_args()


def plot_three_panel(
    rgb: np.ndarray,
    ground_truth: np.ndarray | None,
    predicted_relative: np.ndarray,
    save_path: Path,
) -> None:
    """Plots and saves the 3-panel visualization:
    RGB satellite image | ground-truth height | predicted relative height.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 3 if ground_truth is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))

    # Panel 1: RGB Satellite Image
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Satellite Image")
    axes[0].axis("off")

    if ground_truth is not None:
        # Panel 2: Ground-Truth Height (m)
        im_gt = axes[1].imshow(ground_truth, cmap="terrain")
        axes[1].set_title("Ground-Truth Height (m)")
        axes[1].axis("off")
        fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Predicted Relative Height Map
        im_pred = axes[2].imshow(predicted_relative, cmap="magma")
        axes[2].set_title("Predicted Relative Height ĥ")
        axes[2].axis("off")
        fig.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        # If no ground truth, 2nd panel is predicted relative height
        im_pred = axes[1].imshow(predicted_relative, cmap="magma")
        axes[1].set_title("Predicted Relative Height ĥ")
        axes[1].axis("off")
        fig.colorbar(im_pred, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"Device: {device} | Model size: {args.size}")

    cfg = M1Config(size=args.size, freeze_backbone=True)
    net = HeightPriorNetwork(cfg).to(device)

    # Load weights if checkpoint exists
    if args.checkpoint.exists():
        print(f"Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        net.load_state_dict(state_dict)
    else:
        print(f"Notice: Checkpoint {args.checkpoint} not found. Running with pretrained backbone initialization.")

    net.eval()

    # Load or generate input
    if args.image is not None and args.image.exists():
        pil_img = Image.open(args.image).convert("RGB")
        rgb_np = np.array(pil_img)
        gt_np = np.load(args.height) if args.height and args.height.exists() else None
        gsd = args.gsd
    else:
        print("No input image specified; generating a synthetic validation tile...")
        tile = make_city_tile(gsd_m=args.gsd, seed=777)
        pil_img = tile_to_pil(tile)
        rgb_np = tile.rgb
        gt_np = tile.height_m
        gsd = tile.gsd_m

    with torch.no_grad():
        out = net.infer_pil(pil_img, gsd_m=gsd)
        pred = out.relative[0].cpu().numpy()

    # Match target dimensions
    target_h, target_w = rgb_np.shape[:2]
    if pred.shape != (target_h, target_w):
        pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred_t, size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze().numpy()

    plot_three_panel(
        rgb=rgb_np,
        ground_truth=gt_np,
        predicted_relative=pred,
        save_path=args.out,
    )
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
