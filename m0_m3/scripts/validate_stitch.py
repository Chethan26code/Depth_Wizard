"""
Quick stitching sanity checks - run after m3_stitch.py
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate_stitch(tiles_dir: str, dsm_path: str):
    tiles_dir = Path(tiles_dir)
    dsm_path = Path(dsm_path)

    with open(tiles_dir / "manifest.json") as f:
        manifest = json.load(f)

    tile_size = manifest["tile_size"]

    try:
        import rasterio
        with rasterio.open(dsm_path) as src:
            dsm = src.read(1)
    except Exception:
        dsm = np.load(dsm_path.with_suffix(".npy"))

    expected_h = manifest["tiling_info"]["original_height"]
    expected_w = manifest["tiling_info"]["original_width"]

    print(f"Shape check: got {dsm.shape}, expected ({expected_h}, {expected_w})")
    assert dsm.shape == (expected_h, expected_w), "SHAPE MISMATCH - offset math is likely wrong"

    valid = ~np.isnan(dsm)
    print(f"Valid coverage: {valid.mean():.1%}")
    print(f"Value range (valid pixels): {np.nanmin(dsm):.2f} to {np.nanmax(dsm):.2f}")

    # Seam check: compare pixel differences AT tile boundaries vs just inside them
    boundary_jumps = []
    interior_jumps = []
    n_tile_rows = expected_h // tile_size
    n_tile_cols = expected_w // tile_size

    for r in range(1, n_tile_rows):
        boundary_row = r * tile_size
        if boundary_row >= dsm.shape[0]:
            continue
        above = dsm[boundary_row - 1, :]
        below = dsm[boundary_row, :]
        valid_pair = ~np.isnan(above) & ~np.isnan(below)
        if valid_pair.any():
            boundary_jumps.extend(np.abs(above[valid_pair] - below[valid_pair]))

        interior_row = boundary_row - tile_size // 2
        if 1 <= interior_row < dsm.shape[0]:
            a = dsm[interior_row - 1, :]
            b = dsm[interior_row, :]
            valid_pair2 = ~np.isnan(a) & ~np.isnan(b)
            if valid_pair2.any():
                interior_jumps.extend(np.abs(a[valid_pair2] - b[valid_pair2]))

    if boundary_jumps and interior_jumps:
        avg_boundary = np.mean(boundary_jumps)
        avg_interior = np.mean(interior_jumps)
        print(f"\nAvg pixel jump AT tile boundaries: {avg_boundary:.3f}")
        print(f"Avg pixel jump WITHIN tiles (baseline): {avg_interior:.3f}")
        ratio = avg_boundary / (avg_interior + 1e-6)
        print(f"Ratio: {ratio:.2f}x")
        if ratio > 3:
            print("  -> Significant seam discontinuity at tile boundaries. "
                  "Expected once real per-tile model output is used - flag for overlap/blending later.")
        else:
            print("  -> Boundaries look consistent with interior noise levels. Good.")

    # Visual diagnostic: overlay tile grid lines on the preview
    preview_path = dsm_path.parent / f"{dsm_path.stem}_gridcheck.png"
    valid_values = dsm[valid]
    if valid_values.size:
        low, high = valid_values.min(), valid_values.max()
        norm = np.zeros_like(dsm)
        norm[valid] = (dsm[valid] - low) / (high - low + 1e-8)
        rgb = (np.stack([norm]*3, axis=-1) * 255).astype(np.uint8)
        rgb[~valid] = [255, 0, 0]  # invalid pixels shown in red, impossible to miss

        for r in range(1, n_tile_rows):
            y = r * tile_size
            if y < rgb.shape[0]:
                rgb[y-1:y+1, :] = [0, 255, 0]  # green line at each tile boundary
        for c in range(1, n_tile_cols):
            x = c * tile_size
            if x < rgb.shape[1]:
                rgb[:, x-1:x+1] = [0, 255, 0]

        Image.fromarray(rgb).save(preview_path)
        print(f"\nWrote grid-overlay diagnostic: {preview_path}")
        print("  Green lines = tile boundaries. Red = invalid/NaN. "
              "Look for visible 'steps' in brightness right along green lines.")


if __name__ == "__main__":
    validate_stitch(
        tiles_dir="data/tiles/test_area_tif",
        dsm_path="data/output/test_area_dsm.tif",
    )