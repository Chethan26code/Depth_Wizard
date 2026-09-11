"""
M3 - Assembly / Stitching
Takes per-tile height maps (from M1+M2) plus the M0 manifest.json and
per-tile valid-data masks, and reassembles them into one seamless DSM.
Invalid (nodata/padding) regions are written as NaN, not 0, so they
don't silently corrupt downstream accuracy metrics or the 3D mesh.
"""

import json
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_manifest(tiles_dir: Path) -> dict:
    with open(tiles_dir / "manifest.json") as f:
        return json.load(f)


def make_dummy_height_tiles(tiles_dir: Path, height_tiles_dir: Path):
    """
    TESTING ONLY - fake height data from RGB brightness, so you can test
    stitching before Chethan/Mohona's real output exists. Delete this
    call once real height tiles exist.
    """
    from PIL import Image

    height_tiles_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(tiles_dir)

    for tile in manifest["tiles"]:
        rgb = np.array(Image.open(tiles_dir / tile["filename"]).convert("L"))
        fake_height = rgb.astype(np.float32) / 255.0 * 50.0
        out_name = tile["filename"].replace(".png", ".npy")
        np.save(height_tiles_dir / out_name, fake_height)

    print(f"  Wrote {len(manifest['tiles'])} dummy height tiles to {height_tiles_dir}")


def stitch_tiles(tiles_dir: str, height_tiles_dir: str, output_path: str):
    tiles_dir = Path(tiles_dir)
    height_tiles_dir = Path(height_tiles_dir)
    output_path = Path(output_path)

    manifest = load_manifest(tiles_dir)
    tile_size = manifest["tile_size"]
    tiling_info = manifest["tiling_info"]

    padded_h = tiling_info["original_height"] + tiling_info["pad_h"]
    padded_w = tiling_info["original_width"] + tiling_info["pad_w"]

    canvas = np.full((padded_h, padded_w), np.nan, dtype=np.float32)
    valid_canvas = np.zeros((padded_h, padded_w), dtype=bool)

    base_transform = None
    base_crs = manifest["source_crs"]

    skipped_tiles = []

    for tile in manifest["tiles"]:
        height_filename = tile["filename"].replace(".png", ".npy")
        height_path = height_tiles_dir / height_filename

        if not height_path.exists():
            raise FileNotFoundError(
                f"Missing height tile: {height_path}. "
                f"Every tile in manifest.json needs a matching height file."
            )

        # Skip tiles that were mostly padding/nodata to begin with -
        # their "height" values are meaningless, not real predictions.
        if tile["valid_fraction"] < 0.5:
            skipped_tiles.append(tile["filename"])
            continue

        tile_height = np.load(height_path)

        mask_path = tiles_dir / tile["mask_filename"]
        tile_mask = np.load(mask_path)

        row_off = tile["pixel_offset_y"]
        col_off = tile["pixel_offset_x"]

        region = canvas[row_off:row_off + tile_size, col_off:col_off + tile_size]
        region_valid = valid_canvas[row_off:row_off + tile_size, col_off:col_off + tile_size]

        # Only write into pixels the mask says are real data
        region[tile_mask] = tile_height[tile_mask]
        region_valid[tile_mask] = True

        if tile["row"] == 0 and tile["col"] == 0 and tile["transform"] is not None:
            base_transform = tile["transform"]

    if skipped_tiles:
        print(f"  Skipped {len(skipped_tiles)} low-validity tiles (kept as NaN): {skipped_tiles}")

    final_h = tiling_info["original_height"]
    final_w = tiling_info["original_width"]
    dsm = canvas[:final_h, :final_w]
    valid_final = valid_canvas[:final_h, :final_w]

    coverage = valid_final.mean()
    print(f"  Final DSM valid-data coverage: {coverage:.1%}")
    if coverage < 0.9:
        print(f"  WARNING: significant gaps in the stitched DSM ({(1-coverage):.1%} missing) - "
              f"check whether that's expected (nodata regions) or a bug.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if base_transform is not None and RASTERIO_AVAILABLE:
        transform = Affine(*base_transform)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=dsm.shape[0],
            width=dsm.shape[1],
            count=1,
            dtype=np.float32,
            crs=base_crs,
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(dsm, 1)
        print(f"  Wrote georeferenced DSM GeoTIFF: {output_path}")
    else:
        np.save(output_path.with_suffix(".npy"), dsm)
        print(f"  Wrote non-georeferenced DSM array: {output_path.with_suffix('.npy')}")

    return dsm, valid_final


def save_preview_png(dsm: np.ndarray, valid_mask: np.ndarray, output_path: str):
    """Quick visual check - NaN/invalid pixels shown as solid black, not garbage colors."""
    from PIL import Image

    valid_values = dsm[valid_mask]
    if valid_values.size == 0:
        print("  No valid data to preview.")
        return

    low, high = valid_values.min(), valid_values.max()
    normalized = np.zeros_like(dsm)
    if high - low > 1e-6:
        normalized[valid_mask] = (dsm[valid_mask] - low) / (high - low)

    preview = (normalized * 255).astype(np.uint8)
    preview[~valid_mask] = 0  # invalid regions rendered as black, not misleading colors

    Image.fromarray(preview).save(output_path)
    print(f"  Wrote preview PNG: {output_path}")


if __name__ == "__main__":
    make_dummy_height_tiles(
        tiles_dir=Path("data/tiles/test_area_tif"),
        height_tiles_dir=Path("data/heights/test_area_tif"),
    )

    dsm, valid_mask = stitch_tiles(
        tiles_dir="data/tiles/test_area_tif",
        height_tiles_dir="data/heights/test_area_tif",
        output_path="data/output/test_area_dsm.tif",
    )

    save_preview_png(dsm, valid_mask, "data/output/test_area_dsm_preview.png")