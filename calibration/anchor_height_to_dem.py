"""
anchor_height_to_dem.py  —  v2 (fixed calibration)

Anchor a relative-height array (Depth Anything V2 + Height Head) to a real-world
SRTM DEM, producing a calibrated DSM GeoTIFF in metres.

WHAT CHANGED FROM v1 AND WHY
----------------------------
v1 did:
        a, b = np.polyfit(h, dem, 1)
        output = a * h + b

That is a regression of h onto the DEM, so mathematically:

        std(output) = |corr(h, dem)| * std(dem)

i.e. the output can never vary more than SRTM does, on any terrain. Buildings
are precisely the part of h that SRTM cannot see, so the regression treats them
as noise and removes them. That is why v1 output a ~6 m range over a 1 km scene.

v2 splits the job by spatial frequency:

        DSM = lowpass(SRTM)  +  scale * highpass(h)

  - SRTM supplies ABSOLUTE ELEVATION (the offset). It is reliable for that.
  - The model supplies STRUCTURE (buildings, trees).
  - `scale` (metres per unit of h) is the one number SRTM cannot provide,
    because it has no building signal to calibrate against.

Best source for `scale` is shadow geometry:
        H     = shadow_length * tan(sun_elevation)
        scale = H / (h_roof - h_ground)
Measure one or two buildings, take the median, pass it with --scale.

Without --scale we estimate it by regressing the SMOOTH components against each
other. That is unbiased by buildings, but it still needs the terrain to have
some relief — on flat ground it will under-scale and the script says so.

NOTE: run_pipeline.py calls the calibration too — apply the same change there.

Requirements:  pip install rasterio requests numpy
Usage:
    python anchor_height_to_dem.py input.tif relative_height.npy output_dsm.tif
    python anchor_height_to_dem.py input.tif relative_height.npy out.tif --scale 38.5
"""

import os
import argparse
import numpy as np
import requests
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_bounds

# move this to an env var before the repo goes public
OPENTOPO_API_KEY_DEFAULT = os.environ.get("OPENTOPO_API_KEY", "")


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
# 5. Low-pass filter — numpy only, no scipy dependency
# ---------------------------------------------------------------------------
def _box_blur(a, r):
    """Separable box blur of radius r, via cumulative sums. Edge-padded."""
    if r < 1:
        return a.astype(np.float64).copy()
    a = a.astype(np.float64)
    pad = np.pad(a, r, mode="edge")

    c = np.cumsum(pad, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1)), c], axis=1)
    out = (c[:, 2 * r + 1:] - c[:, :-(2 * r + 1)]) / (2 * r + 1)

    c = np.cumsum(out, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1])), c], axis=0)
    out = (c[2 * r + 1:, :] - c[:-(2 * r + 1), :]) / (2 * r + 1)
    return out


def lowpass(a, radius, passes=2):
    """Two box passes approximate a Gaussian closely enough for this."""
    out = a
    for _ in range(passes):
        out = _box_blur(out, radius)
    return out


# ---------------------------------------------------------------------------
# 6. Estimate `scale` (metres per unit of h) when not supplied
# ---------------------------------------------------------------------------
def estimate_scale(h, dem, radius):
    """
    Regress the SMOOTH part of h against the SMOOTH part of the DEM.

    Using the smooth components means buildings don't corrupt the fit — we are
    only asking "how many metres is one unit of h, judging by the terrain?".

    This still needs the terrain to have relief. If the DEM is nearly flat over
    the scene, the slope is poorly constrained and will come out too small.
    """
    hl = lowpass(h, radius)
    dl = lowpass(dem, radius)

    m = np.isfinite(hl) & np.isfinite(dl)
    x, y = hl[m], dl[m]

    if x.size < 100 or np.std(x) < 1e-9:
        raise ValueError("Not enough valid pixels to estimate scale.")

    a = np.polyfit(x, y, 1)[0]
    r = float(np.corrcoef(x, y)[0, 1])

    print(f"  estimated scale : {a:.3f} m per unit of h")
    print(f"  terrain corr    : {r:+.3f}")
    print(f"  DEM relief      : {float(dl.max() - dl.min()):.1f} m across the scene")

    if abs(dl.max() - dl.min()) < 15:
        print("  WARNING: terrain is nearly flat, so this scale is unreliable.")
        print("           Measure a building shadow and pass --scale instead:")
        print("             H     = shadow_length * tan(sun_elevation)")
        print("             scale = H / (h_roof - h_ground)")
    return abs(a)


# ---------------------------------------------------------------------------
# 7. Calibrate: absolute elevation from SRTM + structure from the model
# ---------------------------------------------------------------------------
def calibrate_dsm(h, dem, scale, radius):
    terrain = lowpass(dem, radius)          # absolute elevation, from SRTM
    detail = h - lowpass(h, radius)         # buildings and trees, from the model
    dsm = terrain + scale * detail

    print("\nResult")
    print(f"  SRTM range      : {float(dem.min()):.1f} to {float(dem.max()):.1f} m"
          f"   (spread {float(dem.max() - dem.min()):.1f} m)")
    print(f"  DSM range       : {float(dsm.min()):.1f} to {float(dsm.max()):.1f} m"
          f"   (spread {float(dsm.max() - dsm.min()):.1f} m)")
    print(f"  detail added    : {float(scale * (detail.max() - detail.min())):.1f} m"
          f"   <- this is what SRTM alone cannot give you")

    if dsm.max() - dsm.min() <= (dem.max() - dem.min()) * 1.05:
        print("  WARNING: DSM spread is no larger than SRTM's. The model is not")
        print("           contributing. Check `scale` and the relative field.")
    return dsm.astype(np.float32)


# ---------------------------------------------------------------------------
# 8. Write calibrated DSM GeoTIFF
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
    p = argparse.ArgumentParser(
        description="Anchor a relative-height .npy to SRTM -> calibrated DSM GeoTIFF."
    )
    p.add_argument("input_tif", help="Input GeoTIFF (defines CRS/bounds/grid)")
    p.add_argument("relative_height_npy", help="Relative height .npy, same H,W as input_tif")
    p.add_argument("output_tif", help="Output calibrated DSM GeoTIFF")
    p.add_argument("--api-key", default=OPENTOPO_API_KEY_DEFAULT,
                   help="OpenTopography API key (or set OPENTOPO_API_KEY)")
    p.add_argument("--dem-type", default="SRTMGL1", help="SRTM product (default SRTMGL1, 30 m)")
    p.add_argument("--scale", type=float, default=None,
                   help="Metres per unit of h. From a shadow: H / (h_roof - h_ground). "
                        "Strongly preferred over the automatic estimate.")
    p.add_argument("--radius", type=int, default=None,
                   help="Low-pass radius in PIXELS separating terrain from structures. "
                        "MUST be roughly 2x the widest building you want to keep, or the "
                        "filter absorbs the building and flattens it. Default: ~60 m worth "
                        "of pixels, which preserves structures up to about 30 m across.")
    args = p.parse_args()

    crs, bounds, transform, shape = read_geotiff_info(args.input_tif)
    latlon_bounds = bounds_to_latlon(crs, bounds)

    h = np.load(args.relative_height_npy).astype(np.float64)
    if h.shape != shape:
        raise ValueError(
            f"Shape mismatch: GeoTIFF is {shape} (H,W) but .npy is {h.shape}. "
            "They must be pixel-aligned."
        )

    # pixel size in metres, for choosing the low-pass radius
    px_m = abs(transform.a)
    if crs.to_epsg() == 4326:          # degrees -> rough metres
        px_m *= 111_320.0
    radius = args.radius if args.radius else max(3, int(round(60.0 / max(px_m, 1e-6))))
    print(f"Pixel size {px_m:.2f} m  ->  low-pass radius {radius} px "
          f"(~{radius * px_m:.0f} m cutoff)")

    raw_dem_path = args.output_tif + "_srtm_raw_tmp.tif"
    fetch_srtm_dem(latlon_bounds, args.api_key, args.dem_type, raw_dem_path)
    dem, dst_transform = resample_dem_to_grid(raw_dem_path, crs, bounds, shape)
    dem = dem.astype(np.float64)

    # clean sentinels so they don't poison the filters
    bad = ~np.isfinite(dem) | (dem < -1e4) | (dem > 1e5)
    if bad.any():
        dem[bad] = np.nanmedian(dem[~bad])
        print(f"Filled {int(bad.sum())} nodata cells in the DEM.")
    h = np.nan_to_num(h, nan=float(np.nanmedian(h)))

    print("\nScale")
    if args.scale is not None:
        scale = args.scale
        print(f"  using supplied scale: {scale:.3f} m per unit of h")
    else:
        scale = estimate_scale(h, dem, radius)

    dsm = calibrate_dsm(h, dem, scale, radius)

    write_calibrated_dsm(dsm, crs, dst_transform, args.output_tif)
    os.remove(raw_dem_path)
    print(f"\nCalibrated DSM saved to: {args.output_tif}")


if __name__ == "__main__":
    main()
