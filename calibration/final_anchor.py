"""
anchor_height_to_dem.py  (v2)

Anchor a relative-height array (Depth Anything V2 + Height Head) to a real-world
SRTM DEM, producing a calibrated DSM GeoTIFF in metres.

WHAT CHANGED FROM v1 AND WHY
----------------------------
v1 fitted  real_elevation = a * h + b  by least squares against SRTM.
That is invalid: SRTM measures ABSOLUTE GROUND TOPOGRAPHY, while h measures
HEIGHT ABOVE GROUND (buildings, trees). Regressing one onto the other forces h
to explain terrain relief it contains no information about. The slope blows up,
the inflated slope multiplies per-pixel noise in h, and you get needle spikes
and negative elevations.

v2 uses the physically correct model:

    DSM = DEM_ground + s * (h - h_ground)

  DEM_ground : resampled SRTM, used DIRECTLY as the base surface (not regressed)
  h_ground   : local ground level inside h, estimated as a low percentile over a
               sliding window (a building cannot fill a whole window)
  s          : metres per unit of h. SRTM CANNOT GIVE YOU THIS. Supply it via
               --scale, or derive it once with --calib (see below) and reuse it.

Also fixed:
  * SRTM voids (-32768) are masked. np.isfinite() does NOT catch them.
  * src_nodata/dst_nodata passed to reproject so voids don't bilinear-smear.
  * Destination array is np.full(nan), not np.empty (uninitialised memory).
  * dst_transform taken from the source GeoTIFF instead of rebuilt from bounds.

VERTICAL DATUM NOTE
-------------------
SRTM elevations are relative to the EGM96 geoid, not the WGS84 ellipsoid. If a
downstream consumer expects ellipsoidal heights, apply a geoid separation
correction; over a single 1 km tile it is a near-constant offset.

Requirements:
    pip install rasterio requests numpy

Usage:
    # If you already know the metric scale of h:
    python anchor_height_to_dem.py in.tif h.npy out_dsm.tif --scale 1.0

    # If you don't: measure one building you know the height of, then
    # (row, col, true_height_in_m):
    python anchor_height_to_dem.py in.tif h.npy out_dsm.tif --calib 412 830 48.0
"""

import os
import argparse

import numpy as np
import requests
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling

SRTM_VOID = -32768.0


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
        return tuple(bounds)
    return transform_bounds(crs, "EPSG:4326", *bounds)


# ---------------------------------------------------------------------------
# 3. Fetch the matching SRTM DEM via the OpenTopography API
# ---------------------------------------------------------------------------
def fetch_srtm_dem(latlon_bounds, api_key, dem_type, out_path, pad_deg=0.005):
    """Pad the request slightly so bilinear resampling has valid neighbours
    at the tile edges instead of falling off the end of the DEM."""
    west, south, east, north = latlon_bounds
    params = {
        "demtype": dem_type,
        "south": south - pad_deg,
        "north": north + pad_deg,
        "west": west - pad_deg,
        "east": east + pad_deg,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    response = requests.get(
        "https://portal.opentopography.org/API/globaldem",
        params=params,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_path


# ---------------------------------------------------------------------------
# 4. Resample DEM onto the exact pixel grid of the input GeoTIFF / h array
# ---------------------------------------------------------------------------
def resample_dem_to_grid(dem_path, dst_crs, dst_transform, dst_shape):
    height, width = dst_shape

    # np.full(nan), NOT np.empty: any pixel the warp does not touch must be
    # identifiably invalid rather than uninitialised memory.
    dem_resampled = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(dem_path) as src:
        src_nodata = src.nodata if src.nodata is not None else SRTM_VOID
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_resampled,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    # Belt and braces: anything still at or near the void sentinel is invalid.
    dem_resampled[dem_resampled <= SRTM_VOID + 1] = np.nan

    n_bad = int(np.isnan(dem_resampled).sum())
    if n_bad:
        pct = 100.0 * n_bad / dem_resampled.size
        print(f"[warn] {n_bad} DEM pixels ({pct:.2f}%) are void/invalid; "
              "they will be filled from the valid median.")
        dem_resampled = np.where(
            np.isnan(dem_resampled),
            np.nanmedian(dem_resampled),
            dem_resampled,
        )
    return dem_resampled


# ---------------------------------------------------------------------------
# 5. Estimate local ground level inside the relative-height array
# ---------------------------------------------------------------------------
def _upsample_bilinear(small, out_shape):
    sh, sw = small.shape
    H, W = out_shape
    if sh == 1 and sw == 1:
        return np.full(out_shape, small[0, 0], dtype=np.float32)

    y = np.linspace(0, sh - 1, H)
    x = np.linspace(0, sw - 1, W)
    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (y - y0)[:, None].astype(np.float32)
    wx = (x - x0)[None, :].astype(np.float32)

    top = small[np.ix_(y0, x0)] * (1 - wx) + small[np.ix_(y0, x1)] * wx
    bot = small[np.ix_(y1, x0)] * (1 - wx) + small[np.ix_(y1, x1)] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def estimate_ground(h, block_px, percentile=5.0):
    """Low-percentile filter over non-overlapping blocks, then bilinear
    upsample. The window must be wider than the largest building footprint,
    otherwise the 'ground' estimate climbs onto the roof."""
    H, W = h.shape
    block_px = max(4, int(block_px))
    nby = int(np.ceil(H / block_px))
    nbx = int(np.ceil(W / block_px))

    padded = np.full((nby * block_px, nbx * block_px),
                     np.nan, dtype=np.float32)
    padded[:H, :W] = h

    blocks = padded.reshape(nby, block_px, nbx, block_px).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(nby, nbx, -1)

    with np.errstate(all="ignore"):
        coarse = np.nanpercentile(
            blocks, percentile, axis=2).astype(np.float32)

    if np.isnan(coarse).any():
        coarse = np.where(np.isnan(coarse), np.nanmedian(coarse), coarse)

    return _upsample_bilinear(coarse, (H, W))


# ---------------------------------------------------------------------------
# 6. Diagnostics — run these before trusting any output
# ---------------------------------------------------------------------------
def report_diagnostics(h, dem):
    valid = np.isfinite(h) & np.isfinite(dem)
    x = h[valid].astype(np.float64)
    y = dem[valid].astype(np.float64)

    print("\n--- diagnostics ---")
    print(f"h    : min={x.min():10.3f}  max={x.max():10.3f}  "
          f"mean={x.mean():10.3f}  std={x.std():8.3f}")
    print(f"DEM  : min={y.min():10.3f}  max={y.max():10.3f}  "
          f"mean={y.mean():10.3f}  std={y.std():8.3f}  (metres)")

    if x.std() > 0 and y.std() > 0:
        r = float(np.corrcoef(x, y)[0, 1])
        print(f"corr(h, DEM) = {r:+.4f}")
        if abs(r) < 0.30:
            print("  -> As expected: h and the DEM are largely uncorrelated.")
            print("     This is exactly why the v1 linear fit produced garbage.")
    print("-------------------\n")


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
        "compress": "deflate",
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(real_elevation, 1)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Anchor a relative-height .npy array to an SRTM DEM "
                    "-> calibrated DSM GeoTIFF."
    )
    parser.add_argument(
        "input_tif", help="Input GeoTIFF (defines CRS/bounds/grid)")
    parser.add_argument("relative_height_npy",
                        help="Relative height .npy (same H,W)")
    parser.add_argument("output_tif", help="Output calibrated DSM GeoTIFF")

    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENTOPO_API_KEY"),
        help="OpenTopography API key (or set OPENTOPO_API_KEY). No key is "
             "hardcoded in this file on purpose.",
    )
    parser.add_argument("--dem-type", default="SRTMGL1",
                        help="SRTM product (default: SRTMGL1, 30 m)")

    parser.add_argument("--scale", type=float, default=None,
                        help="Metres per unit of h. Mutually exclusive with --calib.")
    parser.add_argument("--calib", nargs=3, type=float, metavar=("ROW", "COL", "HEIGHT_M"),
                        default=None,
                        help="Derive scale from one pixel of known height above ground.")

    parser.add_argument("--window-m", type=float, default=200.0,
                        help="Ground-estimation window in metres (default 200). "
                             "Must exceed the largest building footprint.")
    parser.add_argument("--ground-percentile", type=float, default=5.0,
                        help="Percentile within each window taken as ground (default 5).")
    parser.add_argument("--clamp-negative", action="store_true",
                        help="Clamp h-h_ground to >= 0 (no sub-ground structures).")

    args = parser.parse_args()

    if not args.api_key:
        parser.error(
            "No OpenTopography API key. Pass --api-key or set OPENTOPO_API_KEY.")
    if args.scale is not None and args.calib is not None:
        parser.error("Use --scale or --calib, not both.")

    # 1-2. GeoTIFF info + lat/lon bounds
    crs, bounds, transform, shape = read_geotiff_info(args.input_tif)
    latlon_bounds = bounds_to_latlon(crs, bounds)

    pixel_size = abs(transform.a)
    print(f"Grid {shape[1]} x {shape[0]} @ {pixel_size:.3f} m/px")

    dem_res_m = 30.0 if args.dem_type.upper().startswith("SRTM") else 30.0
    posts_y = (shape[0] * pixel_size) / dem_res_m
    posts_x = (shape[1] * pixel_size) / dem_res_m
    if posts_x < 20 or posts_y < 20:
        print(f"[warn] Tile spans only ~{posts_x:.0f} x {posts_y:.0f} SRTM posts. "
              "The DEM contributes a near-constant base level here — fine for "
              "anchoring, useless for fitting anything.")

    # Load relative height
    h = np.load(args.relative_height_npy).astype(np.float32)
    if h.shape != shape:
        raise ValueError(
            f"Shape mismatch: GeoTIFF is {shape} (H,W) but .npy is {h.shape}. "
            "They must be pixel-aligned."
        )

    # 3. Fetch SRTM
    raw_dem_path = args.output_tif + "_srtm_raw_tmp.tif"
    fetch_srtm_dem(latlon_bounds, args.api_key, args.dem_type, raw_dem_path)

    try:
        # 4. Resample onto the exact source grid
        dem = resample_dem_to_grid(raw_dem_path, crs, transform, shape)

        # Diagnostics
        report_diagnostics(h, dem)

        # 5. Local ground level inside h
        block_px = args.window_m / pixel_size
        h_ground = estimate_ground(h, block_px, args.ground_percentile)
        h_rel = h - h_ground
        if args.clamp_negative:
            h_rel = np.maximum(h_rel, 0.0)

        # 6. Metric scale
        if args.calib is not None:
            r, c, true_h = int(args.calib[0]), int(
                args.calib[1]), args.calib[2]
            denom = float(h_rel[r, c])
            if abs(denom) < 1e-6:
                raise ValueError(
                    f"Calibration pixel ({r},{c}) sits at ground level in h "
                    "(h - h_ground ~ 0). Pick a pixel on a rooftop."
                )
            scale = true_h / denom
            print(f"Calibrated scale from pixel ({r},{c}): "
                  f"{true_h:.2f} m / {denom:.4f} units = {scale:.4f} m/unit")
        elif args.scale is not None:
            scale = args.scale
            print(f"Using supplied scale: {scale:.4f} m/unit")
        else:
            scale = 1.0
            print("[warn] No --scale or --calib given; assuming 1.0 m/unit. "
                  "Object heights in the output are NOT metrically trustworthy.")

        # 7. Compose the DSM
        real_elevation = (dem + scale * h_rel).astype(np.float32)

        above_ground = scale * h_rel
        print(f"Object heights above ground: "
              f"{np.nanmin(above_ground):.2f} to {np.nanmax(above_ground):.2f} m "
              f"(p99 = {np.nanpercentile(above_ground, 99):.2f} m)")
        print(f"Output DSM range: "
              f"{np.nanmin(real_elevation):.2f} to {np.nanmax(real_elevation):.2f} m")

        p99 = float(np.nanpercentile(above_ground, 99))
        if p99 > 0 and np.nanmax(above_ground) > 10 * p99:
            print("[warn] Max object height is >10x the 99th percentile. "
                  "That is a spike artefact in h, not a building — consider "
                  "clipping h before anchoring.")

        write_calibrated_dsm(real_elevation, crs, transform, args.output_tif)
        print(f"Calibrated DSM saved to: {args.output_tif}")

    finally:
        if os.path.exists(raw_dem_path):
            os.remove(raw_dem_path)


if __name__ == "__main__":
    main()
