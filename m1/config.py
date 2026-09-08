from __future__ import annotations

from dataclasses import dataclass


# Hugging Face ids. Always the *relative* checkpoints — never the metric-outdoor
# ones. Those metric heads were fitted to car-camera geometry and are meaningless
# from orbit (blueprint §6.1).
BACKBONES = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",  # ~25M, CPU-friendly
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",  # ViT-L, the submission target
}

CANONICAL_GSDS_M = (0.3, 0.5, 1.0, 2.0, 5.0)
TILE_SIZE = 512


@dataclass
class M1Config:
    """Runtime knobs for the height prior. Change size here, nowhere else."""

    size: str = "small"  # small | base | large
    tile_size: int = TILE_SIZE
    freeze_backbone: bool = True
    # Last N transformer blocks stay trainable if we later add LoRA (Phase 1.5).
    lora_last_n_blocks: int = 0
    lora_rank: int = 16
    film_hidden: int = 64

    @property
    def backbone_id(self) -> str:
        if self.size not in BACKBONES:
            raise ValueError(f"Unknown size {self.size!r}. Choose from {list(BACKBONES)}")
        return BACKBONES[self.size]
