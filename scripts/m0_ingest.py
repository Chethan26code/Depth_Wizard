"""
M0 - Ingest & Geo-Parser
Takes a raw image (GeoTIFF or PNG/JPG), extracts geospatial info if present,
normalizes pixel data appropriately for each type, and cuts it into
512x512 tiles with a metadata manifest + valid-data masks for downstream
modules (M1 depth model, M2 calibration, M3 stitching).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


TILE_SIZE = 512


def is_geotiff(path: Path) -> bool:
    if path.suffix.lower() not in (".tif", ".tiff"):
        return False
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required to read .tif files. Install with: pip install rasterio")
    with rasterio.open(path) as src:
        return src.crs is not None


def compute_gsd(transform, crs):
    gsd_x = abs(transform.a)
    gsd_y = abs(transform.e)
    is_geographic = crs.is_geographic if crs is not None else None
    return {
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "units": "degrees (NOT metres - reproject to a projected CRS before using for calibration)"
                 if is_geographic else "metres",
    }


def utm_crs_for_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    hemisphere = 326 if lat >= 0 else 327
    return f"EPSG:{hemisphere}{zone:02d}"


def percentile_stretch_to_uint8(array: np.ndarray, low_pct=2, high_pct=98) -> np.ndarray:
    """
    Rescale pixel values to 0-255 using percentiles instead of raw min/max,
    so a few extreme outlier pixels don't wash out the whole image's contrast.
    Handles any input bit depth (8-bit, 16-bit, float, etc).
    """
    if array.dtype == np.uint8:
        return array  # already correct range, nothing to do

    stretched = np.zeros_like(array, dtype=np.float32)
    for band in range(array.shape[2]):
        band_data = array[:, :, band].astype(np.float32)
        low, high = np.percentile(band_data, [low_pct, high_pct])
        if high - low < 1e-6:
            stretched[:, :, band] = 0
        else:
            clipped = np.clip(band_data, low, high)
            stretched[:, :, band] = (clipped - low) / (high - low) * 255.0

    return stretched.astype(np.uint8)


def load_geotiff(path: Path):
    with rasterio.open(path) as src:
        nodata_value = src.nodata  # None if not defined in the file

        if src.crs.is_geographic:
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.top + bounds.bottom) / 2
            target_crs = utm_crs_for_lonlat(center_lon, center_lat)
            print(f"  Reprojecting from {src.crs} to {target_crs} (metres)")

            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

            reprojected = np.zeros((src.count, height, width), dtype=src.dtypes[0])
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=reprojected[band_idx - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=nodata_value,
                    dst_nodata=nodata_value,
                )
            raw_array = np.transpose(reprojected, (1, 2, 0))
            final_transform, final_crs = transform, target_crs
        else:
            raw_array = np.transpose(src.read(), (1, 2, 0))
            final_transform, final_crs = src.transform, src.crs.to_string()

        if raw_array.shape[2] >= 3:
            raw_array = raw_array[:, :, :3]

        # Build valid-data mask BEFORE normalizing pixel values,
        # since normalization changes what "0" means.
        if nodata_value is not None:
            valid_mask = ~np.all(raw_array == nodata_value, axis=2)
        else:
            valid_mask = np.ones(raw_array.shape[:2], dtype=bool)

        print(f"  dtype: {raw_array.dtype}, applying percentile stretch")
        array = percentile_stretch_to_uint8(raw_array)

        meta = {
            "crs": final_crs,
            "transform": list(final_transform)[:6],
            "width": array.shape[1],
            "height": array.shape[0],
            "gsd": compute_gsd(Affine(*list(final_transform)[:6]) if not isinstance(final_transform, Affine) else final_transform, rasterio.crs.CRS.from_string(final_crs)),
        }
    return array, meta, valid_mask


def load_plain_image(path: Path):
    img = Image.open(path).convert("RGB")
    array = np.array(img)

    if array.dtype != np.uint8:
        print(f"  Non-standard PNG/JPG dtype ({array.dtype}), applying percentile stretch")
        array = percentile_stretch_to_uint8(array)

    valid_mask = np.ones(array.shape[:2], dtype=bool)  # PNG/JPG has no nodata concept

    meta = {
        "crs": None,
        "transform": None,
        "width": array.shape[1],
        "height": array.shape[0],
        "gsd": None,
    }
    return array, meta, valid_mask


def pad_to_multiple(array: np.ndarray, mask: np.ndarray, tile_size: int):
    h, w = array.shape[:2]
    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size
    if pad_h or pad_w:
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
        # Padded region is explicitly marked invalid - it's not real data
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    return array, mask, pad_h, pad_w


def tile_offset_transform(base_transform_coeffs, col_off, row_off):
    base = Affine(*base_transform_coeffs)
    return base * Affine.translation(col_off, row_off)


def cut_tiles(array: np.ndarray, mask: np.ndarray, meta: dict, tile_size: int = TILE_SIZE):
    padded, padded_mask, pad_h, pad_w = pad_to_multiple(array, mask, tile_size)
    h, w = padded.shape[:2]

    tiles = []
    tile_id = 0
    for row_off in range(0, h, tile_size):
        for col_off in range(0, w, tile_size):
            tile_array = padded[row_off:row_off + tile_size, col_off:col_off + tile_size]
            tile_mask = padded_mask[row_off:row_off + tile_size, col_off:col_off + tile_size]

            valid_fraction = float(tile_mask.mean())

            tile_meta = {
                "tile_id": tile_id,
                "row": row_off // tile_size,
                "col": col_off // tile_size,
                "pixel_offset_x": col_off,
                "pixel_offset_y": row_off,
                "tile_size": tile_size,
                "is_edge_tile": (row_off + tile_size > h - pad_h) or (col_off + tile_size > w - pad_w),
                "valid_fraction": round(valid_fraction, 4),
            }

            if meta["transform"] is not None:
                tile_transform = tile_offset_transform(meta["transform"], col_off, row_off)
                tile_meta["transform"] = list(tile_transform)[:6]
                tile_meta["crs"] = meta["crs"]
            else:
                tile_meta["transform"] = None
                tile_meta["crs"] = None

            tiles.append((tile_array, tile_mask, tile_meta))
            tile_id += 1

    tiling_info = {
        "original_height": h - pad_h,
        "original_width": w - pad_w,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }
    return tiles, tiling_info


def process_image(input_path: str, output_dir: str, tile_size: int = TILE_SIZE):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    georeferenced = is_geotiff(input_path)
    print(f"Processing {input_path.name} | georeferenced: {georeferenced}")

    if georeferenced:
        array, meta, valid_mask = load_geotiff(input_path)
    else:
        array, meta, valid_mask = load_plain_image(input_path)

    tiles, tiling_info = cut_tiles(array, valid_mask, meta, tile_size)

    manifest = {
        "source_file": str(input_path),
        "georeferenced": georeferenced,
        "source_crs": meta["crs"],
        "source_gsd": meta["gsd"],
        "original_width": meta["width"],
        "original_height": meta["height"],
        "tiling_info": tiling_info,
        "tile_size": tile_size,
        "tiles": [],
    }

    low_valid_tiles = []
    for tile_array, tile_mask, tile_meta in tiles:
        tile_filename = f"tile_r{tile_meta['row']:03d}_c{tile_meta['col']:03d}.png"
        Image.fromarray(tile_array).save(output_dir / tile_filename)

        mask_filename = tile_filename.replace(".png", "_mask.npy")
        np.save(output_dir / mask_filename, tile_mask)

        tile_meta["filename"] = tile_filename
        tile_meta["mask_filename"] = mask_filename
        manifest["tiles"].append(tile_meta)

        if tile_meta["valid_fraction"] < 0.5:
            low_valid_tiles.append(tile_filename)

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  -> {len(tiles)} tiles written to {output_dir}")
    print(f"  -> manifest: {manifest_path}")
    if low_valid_tiles:
        print(f"  WARNING: {len(low_valid_tiles)} tiles are <50% valid data: {low_valid_tiles}")
        print(f"  Flag these to Chethan/Mohona - they should probably be skipped or down-weighted.")
    return manifest


if __name__ == "__main__":
    process_image("data/raw/TEST1US/test_area.tif", "data/tiles/test_area_tif")
    process_image("data/raw/TEST1US/test_area.png", "data/tiles/test_area_png")