"""Tiny city-like RGB + nDSM tiles so you can train without the 80 GB GAMUS dump.

This is *not* a substitute for GAMUS / US3D. It exists so you can:
  1. understand scale ambiguity on a surface you fully control
  2. debug the training loop on CPU in minutes
  3. keep working while the real dataset downloads
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SyntheticTile:
    rgb: np.ndarray  # uint8 HWC
    height_m: np.ndarray  # float32 HW, nDSM metres
    gsd_m: float
    building_count: int


def make_city_tile(
    size: int = 512,
    gsd_m: float = 1.0,
    n_buildings: int | None = None,
    seed: int | None = None,
) -> SyntheticTile:
    rng = np.random.default_rng(seed)
    if n_buildings is None:
        n_buildings = int(rng.integers(8, 25))

    # Ground: noisy green. Buildings: grey blocks whose brightness tracks height
    # a little — the network can cheat on this toy data, which is fine for a
    # plumbing test and *not* fine as a real benchmark.
    grass = np.array([34, 92, 48], dtype=np.float32)
    rgb = grass + rng.normal(0, 8, size=(size, size, 3))
    height = rng.normal(0.4, 0.15, size=(size, size)).astype(np.float32)
    height = np.clip(height, 0.0, None)

    for _ in range(n_buildings):
        w = int(rng.integers(16, 90))
        h = int(rng.integers(16, 90))
        x = int(rng.integers(0, size - w))
        y = int(rng.integers(0, size - h))
        storeys = int(rng.integers(2, 28))
        z = storeys * 3.2 + float(rng.normal(0, 0.4))
        height[y : y + h, x : x + w] = z
        shade = 70 + np.clip(z, 0, 80) * 1.6
        colour = np.array([shade, shade + 4, shade + 10], dtype=np.float32)
        rgb[y : y + h, x : x + w] = colour
        # Crude shadow south-east of the footprint, length ∝ height.
        # Real M2 uses sun metadata; here it is just a visual cue.
        shadow_px = max(2, int(z / (gsd_m * 8)))
        ys = min(size, y + h + shadow_px)
        xs = min(size, x + w + shadow_px // 2)
        rgb[y + h : ys, x : xs] *= 0.45

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return SyntheticTile(rgb=rgb, height_m=height, gsd_m=float(gsd_m), building_count=n_buildings)


def tile_to_pil(tile: SyntheticTile) -> Image.Image:
    return Image.fromarray(tile.rgb, mode="RGB")
