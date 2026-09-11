"""
M0: Preprocessing
Takes a raw image (GeoTIFF or PNG/JPG) - either a file path (local
testing) or raw bytes (from app upload) - detects type,
reprojects to metric UTM if needed, normalizes pixel values, builds a
valid-data mask, and returns everything processing needs to run depth
estimation. No tiling, no file writes required for the app path.
"""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


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
        "units": "degrees (NOT metres)" if is_geographic else "metres",
    }


def percentile_stretch_to_uint8(array: np.ndarray, low_pct=2, high_pct=98) -> np.ndarray:
    """Rescale to 0-255 using percentiles so outlier pixels don't wash out contrast."""
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


def _is_geotiff(filename: str, has_crs_check) -> bool:
    if not filename.lower().endswith((".tif", ".tiff")):
        return False
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio is required for .tif files. Install with: pip install rasterio")
    return has_crs_check()


def _process_geotiff(open_dataset):
    """open_dataset is a context manager yielding a rasterio dataset. works
    the same whether it came from a file path or an in-memory upload."""
    with open_dataset() as src:
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


def _process_plain_image(pil_image: Image.Image):
    img = pil_image.convert("RGB")
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


# ---------- Main entry point: use this from app ----------

def preprocess_image(file_bytes: bytes, filename: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    THE function Maansi's app calls. Give it the raw uploaded file's bytes
    and its filename (used only to check extension). Returns:
      - rgb_array: (H, W, 3) uint8 numpy array, ready for Chethan's model
      - valid_mask: (H, W) bool numpy array, True = real data
      - metadata: dict with georeferenced/crs/transform/gsd info for Mohona

    No files written to disk - everything stays in memory.
    """
    is_tif = filename.lower().endswith((".tif", ".tiff"))

    if is_tif:
        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio is required for .tif files. Install with: pip install rasterio")

        def open_dataset():
            memfile = MemoryFile(file_bytes)
            return memfile.open()

        with MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                georeferenced = src.crs is not None

        if georeferenced:
            array, valid_mask, meta = _process_geotiff(lambda: MemoryFile(file_bytes).open())
        else:
            # a .tif with no CRS - treat like a plain image
            with MemoryFile(file_bytes) as memfile:
                with memfile.open() as src:
                    raw = np.transpose(src.read(), (1, 2, 0))
                    if raw.shape[2] >= 3:
                        raw = raw[:, :, :3]
            pil_img = Image.fromarray(raw)
            array, valid_mask, meta = _process_plain_image(pil_img)
    else:
        pil_img = Image.open(io.BytesIO(file_bytes))
        array, valid_mask, meta = _process_plain_image(pil_img)

    return array, valid_mask, meta


# ---------- Local testing helpers (file-based, to run/debug) ----------

def preprocess_from_path(input_path: str, output_dir: str = None):
    """
    Local testing wrapper: reads a file from disk, runs preprocess_image,
    optionally saves results to disk so you can inspect them.
    """
    input_path = Path(input_path)
    with open(input_path, "rb") as f:
        file_bytes = f.read()

    print(f"Processing {input_path.name}")
    array, valid_mask, meta = preprocess_image(file_bytes, input_path.name)

    valid_fraction = float(valid_mask.mean())
    print(f"  Array shape: {array.shape}, dtype: {array.dtype}")
    print(f"  Georeferenced: {meta['georeferenced']}")
    if meta['georeferenced']:
        print(f"  CRS: {meta['crs']}")
        print(f"  GSD: {meta['gsd']}")
    print(f"  Valid data: {valid_fraction:.1%}")

    if valid_fraction < 0.9:
        print(f"  NOTE: {(1 - valid_fraction):.1%} of the image is nodata/invalid.")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "preprocessed_rgb.npy", array)
        np.save(output_dir / "valid_mask.npy", valid_mask)
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Saved to {output_dir}/")

    return array, valid_mask, meta


if __name__ == "__main__":
    # Paths are relative to THIS script's location, not wherever it runs from
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent  # adjust if folder depth differs

    preprocess_from_path(
        PROJECT_ROOT / "data" / "raw" / "TEST1US" / "test_area.tif",
        PROJECT_ROOT / "data" / "preprocessed" / "test_area_tif",
    )
    preprocess_from_path(
        PROJECT_ROOT / "data" / "raw" / "TEST1US" / "test_area.png",
        PROJECT_ROOT / "data" / "preprocessed" / "test_area_png",
    )
