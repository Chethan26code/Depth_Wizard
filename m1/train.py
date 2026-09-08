"""Train the DPT decoder + FiLM + uncertainty head on synthetic tiles.

This is the Phase-1 *loop*, not the Phase-1 *dataset*. Swap the dataset class
for GAMUS later without touching the model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from m1.config import CANONICAL_GSDS_M, M1Config
from m1.losses import total_m1_loss
from m1.model import HeightPriorNetwork, pick_device
from m1.synthetic import make_city_tile


class SyntheticHeightDataset(Dataset):
    """Synthetic dataset simulating nadir satellite optical tiles + nDSM ground truth.

    - `image`: 3-channel RGB simulated satellite tile.
    - `height`: single-channel pixel-wise ground-truth height above terrain (nDSM).
    - `gsd`: Ground Sampling Distance (meters per pixel), used for FiLM conditioning.
    """

    def __init__(self, n: int = 32, tile_size: int = 512, seed: int = 0):
        self.n = n
        self.tile_size = tile_size
        self.seed = seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng_gsd = CANONICAL_GSDS_M[idx % len(CANONICAL_GSDS_M)]
        tile = make_city_tile(size=self.tile_size, gsd_m=rng_gsd, seed=self.seed + idx)
        # SILog requires strictly positive values; add 0.01m so log() is defined at bare ground.
        height = torch.from_numpy(tile.height_m).float() + 0.01
        image = Image.fromarray(tile.rgb)
        return {
            "image": image,
            "rgb_np": tile.rgb,
            "height": height,
            "height_np": tile.height_m,
            "gsd": torch.tensor(tile.gsd_m, dtype=torch.float32),
        }


def collate(batch):
    return {
        "images": [b["image"] for b in batch],
        "rgb_np": [b["rgb_np"] for b in batch],
        "height": torch.stack([b["height"] for b in batch]),
        "height_np": [b["height_np"] for b in batch],
        "gsd": torch.stack([b["gsd"] for b in batch]),
    }


def save_prediction_visualization(
    rgb: np.ndarray,
    ground_truth: np.ndarray,
    predicted_relative: np.ndarray,
    save_path: Path,
    title_suffix: str = "",
) -> None:
    """Saves a 3-panel side-by-side visualization:
    RGB satellite image | ground-truth height | predicted relative height.
    """
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: RGB Satellite Image
    axes[0].imshow(rgb)
    axes[0].set_title("RGB Satellite Image")
    axes[0].axis("off")

    # Panel 2: Ground-Truth Height (nDSM in meters)
    im_gt = axes[1].imshow(ground_truth, cmap="terrain")
    axes[1].set_title("Ground-Truth Height (m)")
    axes[1].axis("off")
    fig.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Predicted Relative Height Map (ĥ)
    im_pred = axes[2].imshow(predicted_relative, cmap="magma")
    axes[2].set_title(f"Predicted Relative Height ĥ {title_suffix}".strip())
    axes[2].axis("off")
    fig.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"Visualization saved to: {save_path}")


@torch.no_grad()
def evaluate(
    net: HeightPriorNetwork,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Runs evaluation over the validation set."""
    net.eval()
    val_losses = []
    val_silog = []
    val_grad = []
    val_nll = []

    for batch in loader:
        pv = net.preprocess(batch["images"], device)
        gsd = batch["gsd"].to(device)
        target = batch["height"].to(device)

        out = net(pv, gsd)
        pred = F.interpolate(
            out.relative.unsqueeze(1),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        log_var = F.interpolate(
            out.log_var.unsqueeze(1),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        pred_pos = pred - pred.amin(dim=(-2, -1), keepdim=True) + 0.01
        loss, parts = total_m1_loss(pred_pos, log_var, target)

        val_losses.append(parts["total"])
        val_silog.append(parts["silog"])
        val_grad.append(parts["grad"])
        val_nll.append(parts["nll"])

    net.train()
    return {
        "val_total": float(np.mean(val_losses)),
        "val_silog": float(np.mean(val_silog)),
        "val_grad": float(np.mean(val_grad)),
        "val_nll": float(np.mean(val_nll)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small", choices=["small", "base", "large"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-tiles", type=int, default=8, help="synthetic tiles for training")
    p.add_argument("--val-tiles", type=int, default=2, help="synthetic tiles for validation")
    p.add_argument("--out", type=Path, default=Path("outputs/checkpoints"))
    p.add_argument("--vis-out", type=Path, default=Path("outputs/visualizations"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.vis_out.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    cfg = M1Config(size=args.size, freeze_backbone=True)
    net = HeightPriorNetwork(cfg).to(device)
    trainable, total = net.trainable_parameter_count()
    print(f"Device: {device} | Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M params")

    opt = torch.optim.AdamW(
        (p for p in net.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=1e-4,
    )

    train_dataset = SyntheticHeightDataset(n=args.train_tiles, seed=100)
    val_dataset = SyntheticHeightDataset(n=args.val_tiles, seed=9000)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    best_val_loss = float("inf")
    print(f"Starting training for {args.epochs} epoch(s)...")

    for epoch in range(args.epochs):
        net.train()
        running_train_losses = []
        for step, batch in enumerate(train_loader):
            pv = net.preprocess(batch["images"], device)
            gsd = batch["gsd"].to(device)
            target = batch["height"].to(device)

            out = net(pv, gsd)
            pred = F.interpolate(
                out.relative.unsqueeze(1),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            log_var = F.interpolate(
                out.log_var.unsqueeze(1),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            pred_pos = pred - pred.amin(dim=(-2, -1), keepdim=True) + 0.01
            loss, parts = total_m1_loss(pred_pos, log_var, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            running_train_losses.append(parts["total"])
            if step % 4 == 0:
                print(
                    f"Epoch {epoch:02d} | Step {step:03d} | "
                    f"Loss: {parts['total']:.4f} (SILog: {parts['silog']:.4f}, "
                    f"Grad: {parts['grad']:.4f}, NLL: {parts['nll']:.4f})"
                )

        # Validation phase
        val_metrics = evaluate(net, val_loader, device)
        mean_train_loss = float(np.mean(running_train_losses))
        print(
            f"--> Epoch {epoch:02d} Complete | "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_total']:.4f} (SILog: {val_metrics['val_silog']:.4f}, "
            f"Grad: {val_metrics['val_grad']:.4f})"
        )

        # Checkpoint saving: standard epoch checkpoint
        ckpt_path = args.out / f"m1_{args.size}_epoch{epoch}.pt"
        torch.save(
            {
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "cfg": args.size,
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": val_metrics["val_total"],
            },
            ckpt_path,
        )

        # Best checkpoint saving
        if val_metrics["val_total"] < best_val_loss:
            best_val_loss = val_metrics["val_total"]
            best_ckpt = args.out / f"m1_{args.size}_best.pt"
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": args.size,
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "val_loss": best_val_loss,
                },
                best_ckpt,
            )
            print(f"--> Saved NEW BEST model to: {best_ckpt} (Val Loss: {best_val_loss:.4f})")

    # Generate 3-panel visualization using a sample from the validation set
    print("Generating 3-panel validation visualization...")
    net.eval()
    val_sample = val_dataset[0]
    with torch.no_grad():
        out_vis = net.infer_pil(val_sample["image"], gsd_m=val_sample["gsd"].item())
        pred_map = out_vis.relative[0].cpu().numpy()

    # Interpolate to match original ground truth size if needed
    if pred_map.shape != val_sample["height_np"].shape:
        pred_t = torch.from_numpy(pred_map).unsqueeze(0).unsqueeze(0)
        target_h, target_w = val_sample["height_np"].shape
        pred_map = F.interpolate(pred_t, size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze().numpy()

    vis_file = args.vis_out / f"m1_{args.size}_val_preview.png"
    save_prediction_visualization(
        rgb=val_sample["rgb_np"],
        ground_truth=val_sample["height_np"],
        predicted_relative=pred_map,
        save_path=vis_file,
        title_suffix=f"(Epoch {args.epochs-1})",
    )
    print("M1 training and validation pipeline completed successfully!")


if __name__ == "__main__":
    main()
