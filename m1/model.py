"""Depth Anything V2 wrapped as a *height* prior, not a camera-depth model.

What stays frozen (blueprint Figure 5):
    the DINOv2 / ViT encoder — ~25M (small) to ~300M (large) parameters
    that already understand edges, planes, and occlusion.

What we train:
    the DPT neck + depth head (already in the HF checkpoint), plus a tiny
    GSD FiLM branch and an uncertainty head.

Why freeze: you do not have an A100. Training 14M decoder params overnight
on a T4 (or slowly on CPU with the Small backbone) is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from m1.config import M1Config


@dataclass
class HeightPriorOutput:
    relative: torch.Tensor  # (B, H, W) unitless field ĥ
    log_var: torch.Tensor  # (B, H, W) log σ²


class HeightPriorNetwork(nn.Module):
    def __init__(self, cfg: M1Config | None = None):
        super().__init__()
        self.cfg = cfg or M1Config()
        self.processor = AutoImageProcessor.from_pretrained(self.cfg.backbone_id)
        self.da = AutoModelForDepthEstimation.from_pretrained(self.cfg.backbone_id)

        if self.cfg.freeze_backbone:
            self._freeze_backbone()

        fusion_ch = int(getattr(self.da.config, "fusion_hidden_size", 64))
        self.gsd_mlp = nn.Sequential(
            nn.Linear(1, self.cfg.film_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.film_hidden, fusion_ch * 2),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(fusion_ch, fusion_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fusion_ch // 2, 1, kernel_size=1),
        )

    def _freeze_backbone(self) -> None:
        if hasattr(self.da, "backbone"):
            for p in self.da.backbone.parameters():
                p.requires_grad = False
        else:
            for name, p in self.da.named_parameters():
                if "neck" not in name and "head" not in name:
                    p.requires_grad = False

    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def preprocess(self, images, device: torch.device) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(device)

    def _film(self, feat: torch.Tensor, gsd_m: torch.Tensor) -> torch.Tensor:
        """Feature-wise Linear Modulation: feat' = γ(gsd) * feat + β(gsd).

        A 0.3 m aerial pixel and a 2 m satellite pixel can look similar but
        mean totally different real-world heights. Feeding log(GSD) tells the
        decoder how big a pixel is (blueprint §6.3).
        """
        log_gsd = torch.log(gsd_m.clamp_min(1e-4)).unsqueeze(-1).to(dtype=feat.dtype)
        gb = self.gsd_mlp(log_gsd)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return feat * (1.0 + gamma) + beta

    def _backbone_features(self, pixel_values: torch.Tensor):
        backbone = self.da.backbone
        if hasattr(backbone, "forward_with_filtered_kwargs"):
            out = backbone.forward_with_filtered_kwargs(pixel_values)
        else:
            out = backbone(pixel_values)
        if hasattr(out, "feature_maps") and out.feature_maps is not None:
            maps = out.feature_maps
        elif hasattr(out, "hidden_states") and out.hidden_states is not None:
            maps = out.hidden_states
        else:
            maps = out[0] if isinstance(out, (tuple, list)) else out
        return maps

    def forward(self, pixel_values: torch.Tensor, gsd_m: torch.Tensor) -> HeightPriorOutput:
        _, _, height, width = pixel_values.shape
        patch_size = getattr(self.da.config, "patch_size", 14)
        patch_height = height // patch_size
        patch_width = width // patch_size

        maps = self._backbone_features(pixel_values)
        hidden = self.da.neck(maps, patch_height, patch_width)
        if isinstance(hidden, (list, tuple)):
            hidden = list(hidden)
            idx = int(getattr(self.da.config, "head_in_index", -1))
            hidden[idx] = self._film(hidden[idx], gsd_m)
            fused = hidden[idx]
            predicted = self.da.head(hidden, patch_height, patch_width)
        else:
            fused = self._film(hidden, gsd_m)
            predicted = self.da.head(fused, patch_height, patch_width)

        if isinstance(predicted, (list, tuple)):
            predicted = predicted[0]
        if predicted.dim() == 4:
            predicted = predicted.squeeze(1)

        log_var = self.uncertainty_head(fused)
        log_var = F.interpolate(
            log_var,
            size=predicted.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        return HeightPriorOutput(relative=predicted, log_var=log_var)

    @torch.no_grad()
    def infer_pil(self, image, gsd_m: float = 1.0) -> HeightPriorOutput:
        self.eval()
        device = next(self.parameters()).device
        pv = self.preprocess(image, device)
        gsd = torch.tensor([gsd_m], device=device, dtype=pv.dtype)
        return self(pv, gsd)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
