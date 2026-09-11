"""
M0 - Ingest & Preprocessing (no tiling version)
Takes a raw image (GeoTIFF or PNG/JPG), detects type, reprojects to
metric UTM if the source is in degrees, normalizes pixel values, and
builds a valid-data mask. Outputs ONE array (not tiles) + metadata for
Mohona/Chethan to run on directly.
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


def is_geotiff(path: Path) -> bool:
    if path.suffix.lower() not in (".tif", ".tiff"):
        return False
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required to read .tif files. Install with: pip install rasterio")
    with rasterio.open(path) as src:
        return src.crs is not None


def utm_crs_for_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) / 6) + 1
    hemisphere = 326 if lat >= 0 else 327
    return f"EPSG:{hemisphere}{zone:02d}"


def compute_gsd(transform, crs) -> dict:
    gsd_x = abs(transform.a)
    gsd_y = abs(transform.e)
    is_geographic = crs.is_geographic if crs is not None else None
    return {
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "units": "degrees (NOT metres - not usable for calibration as-is)"
                 if is_geographic else "metres",
    }


def percentile_stretch_to_uint8(array: np.ndarray, low_pct=2, high_pct=98) -> np.ndarray:
    """Rescale to 0-255 using percentiles, so outlier pixels don't wash out contrast."""
    if array.dtype == np.uint8:
        return array
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
        nodata_value = src.nodata

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

        if nodata_value is not None:
            valid_mask = ~np.all(raw_array == nodata_value, axis=2)
        else:
            valid_mask = np.ones(raw_array.shape[:2], dtype=bool)

        print(f"  dtype: {raw_array.dtype}, applying percentile stretch")
        array = percentile_stretch_to_uint8(raw_array)

        transform_obj = final_transform if isinstance(final_transform, Affine) else Affine(*final_transform)
        gsd = compute_gsd(transform_obj, rasterio.crs.CRS.from_string(final_crs))

        meta = {
            "georeferenced": True,
            "crs": final_crs,
            "transform": list(transform_obj)[:6],
            "gsd": gsd,
            "width": array.shape[1],
            "height": array.shape[0],
        }

    return array, valid_mask, meta


def load_plain_image(path: Path):
    img = Image.open(path).convert("RGB")
    array = np.array(img)

    if array.dtype != np.uint8:
        print(f"  Non-standard dtype ({array.dtype}), applying percentile stretch")
        array = percentile_stretch_to_uint8(array)

    valid_mask = np.ones(array.shape[:2], dtype=bool)  # PNG/JPG has no nodata concept

    meta = {
        "georeferenced": False,
        "crs": None,
        "transform": None,
        "gsd": None,
        "width": array.shape[1],
        "height": array.shape[0],
    }
    return array, valid_mask, meta


def process_image(input_path: str, output_dir: str):
    """
    Loads, preprocesses, and saves ONE array + mask + metadata - no tiling.
    This is what gets handed to Mohona/Chethan directly.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    georeferenced = is_geotiff(input_path)
    print(f"Processing {input_path.name} | georeferenced: {georeferenced}")

    if georeferenced:
        array, valid_mask, meta = load_geotiff(input_path)
    else:
        array, valid_mask, meta = load_plain_image(input_path)

    # Save the RGB array (as .npy, so exact pixel values are preserved -
    # not re-compressed like a PNG would)
    array_path = output_dir / "preprocessed_rgb.npy"
    np.save(array_path, array)

    mask_path = output_dir / "valid_mask.npy"
    np.save(mask_path, valid_mask)

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    valid_fraction = float(valid_mask.mean())
    print(f"  Array shape: {array.shape}, valid data: {valid_fraction:.1%}")
    print(f"  -> {array_path}")
    print(f"  -> {mask_path}")
    print(f"  -> {meta_path}")

    if valid_fraction < 0.9:
        print(f"  NOTE: {(1 - valid_fraction):.1%} of the image is nodata/invalid - "
              f"tell Mohona/Chethan to check the mask before running.")

    return array, valid_mask, meta


from pathlib import Path

# This always points to wherever this .py file physically lives,
# regardless of what folder you ran `python` from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # adjust .parent count based on your actual folder depth

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

if __name__ == "__main__":
    process_image(RAW_DIR / "test_area.tif", PREPROCESSED_DIR / "test_area_tif")
    process_image(RAW_DIR / "test_area.png", PREPROCESSED_DIR / "test_area_png")