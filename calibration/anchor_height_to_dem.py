"""
anchor_height_to_dem.py

Anchor a relative-height array (from Depth Anything V2 + Height Head)
to a real-world SRTM DEM, producing a calibrated DSM GeoTIFF in metres.

Works for ANY GeoTIFF + matching relative-height .npy pair, as long as:
  - the .npy array's shape matches the GeoTIFF's (height, width)
  - the .npy array is pixel-aligned with the GeoTIFF (same grid, no offset)

Requirements:
    pip install rasterio requests numpy

Usage:
    python anchor_height_to_dem.py input.tif relative_height.npy output_dsm.tif
"""

import os
import argparse
import numpy as np
import requests
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds

OPENTOPO_API_KEY_DEFAULT = "2abedde1f0675abe37066b9daded3e81"


# ---------------------------------------------------------------------------
# 1. Read the input GeoTIFF's georeferencing info
# ---------------------------------------------------------------------------
def read_geotiff_info(input_path):
    with rasterio.open(input_path) as src:
        return src.crs, src.bounds, src.transform, (src.height, src.width)


# ---------------------------------------------------------------------------
# 2. Reproject bounds to lat/lon (EPSG:4326) for the SRTM API
# ---------------------------------------------------------------------------
def bounds_to_latlon(crs, bounds):
    if crs.to_epsg() == 4326:
        return bounds
    return transform_bounds(crs, "EPSG:4326", *bounds)


# ---------------------------------------------------------------------------
# 3. Fetch the matching SRTM DEM via the OpenTopography API
# ---------------------------------------------------------------------------
def fetch_srtm_dem(latlon_bounds, api_key, dem_type, out_path):
    west, south, east, north = latlon_bounds
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type,
        "south": south, "north": north, "west": west, "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_path


# ---------------------------------------------------------------------------
# 4. Resample DEM onto the same pixel grid as the input GeoTIFF / h array
# ---------------------------------------------------------------------------
def resample_dem_to_grid(dem_path, dst_crs, dst_bounds, dst_shape):
    """Reproject/resample the SRTM DEM so its pixel grid exactly matches
    the GeoTIFF's (and therefore the relative-height array's) grid."""
    height, width = dst_shape
    dst_transform = from_bounds(*dst_bounds, width=width, height=height)

    dem_resampled = np.empty((height, width), dtype=np.float32)

    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )

    return dem_resampled, dst_transform


# ---------------------------------------------------------------------------
# 5. Fit scale + offset: real_elevation = a * h + b
# ---------------------------------------------------------------------------
def fit_anchor_transform(h, dem):
    """
    Linear regression fitting relative height (h) to real DEM elevation.
    Uses all valid pixels; flatten both arrays and fit y = a*x + b.
    """
    valid = np.isfinite(h) & np.isfinite(dem)
    x = h[valid].astype(np.float64)
    y = dem[valid].astype(np.float64)

    a, b = np.polyfit(x, y, 1)  # slope, intercept
    print(f"Fitted transform: real_elevation = {a:.4f} * h + {b:.4f}")

    # Report fit quality
    pred = a * x + b
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    print(f"Fit RMSE: {rmse:.2f} m")

    return a, b


# ---------------------------------------------------------------------------
# 6. Apply transform to get real-world elevation
# ---------------------------------------------------------------------------
def apply_anchor_transform(h, a, b):
    return (a * h + b).astype(np.float32)


# ---------------------------------------------------------------------------
# 7. Write calibrated DSM GeoTIFF
# ---------------------------------------------------------------------------
def write_calibrated_dsm(real_elevation, crs, transform, out_path):
    height, width = real_elevation.shape
    meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": None,
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(real_elevation, 1)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Anchor a relative-height .npy array to SRTM DEM -> calibrated DSM GeoTIFF."
    )
    parser.add_argument("input_tif", help="Path to input GeoTIFF (defines CRS/bounds/grid)")
    parser.add_argument("relative_height_npy", help="Path to relative height .npy (same H,W as input_tif)")
    parser.add_argument("output_tif", help="Path to output calibrated DSM GeoTIFF")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENTOPO_API_KEY", OPENTOPO_API_KEY_DEFAULT),
        help="OpenTopography API key",
    )
    parser.add_argument("--dem-type", default="SRTMGL1", help="SRTM product (default: SRTMGL1, 30m)")
    args = parser.parse_args()

    # 1-2. Read GeoTIFF info + get lat/lon bounds
    crs, bounds, transform, shape = read_geotiff_info(args.input_tif)
    latlon_bounds = bounds_to_latlon(crs, bounds)

    # Load relative height array
    h = np.load(args.relative_height_npy)
    if h.shape != shape:
        raise ValueError(
            f"Shape mismatch: GeoTIFF is {shape} (H,W) but .npy is {h.shape}. "
            "They must be pixel-aligned."
        )

    # 3. Fetch SRTM DEM
    raw_dem_path = args.output_tif + "_srtm_raw_tmp.tif"
    fetch_srtm_dem(latlon_bounds, args.api_key, args.dem_type, raw_dem_path)

    # 4. Resample DEM onto the same grid as h / input GeoTIFF
    dem_resampled, dst_transform = resample_dem_to_grid(raw_dem_path, crs, bounds, shape)

    # 5. Fit anchor transform (h -> real elevation)
    a, b = fit_anchor_transform(h, dem_resampled)

    # 6. Apply transform
    real_elevation = apply_anchor_transform(h, a, b)

    # 7. Write final calibrated DSM
    write_calibrated_dsm(real_elevation, crs, dst_transform, args.output_tif)

    os.remove(raw_dem_path)  # cleanup intermediate SRTM download
    print(f"Calibrated DSM saved to: {args.output_tif}")


if __name__ == "__main__":
    main()
