"""
run_pipeline.py — DepthWizard M1 + DEM Calibration Pipeline (v2: Frequency-Split Calibration)

Connects:
1. M1 relative height model (Depth Anything V2 + Height Head)
2. Frequency-split calibration:
      terrain = lowpass(SRTM)       <- absolute elevation
      detail  = h - lowpass(h)      <- buildings & structures
      dsm     = terrain + scale * detail
3. Handles both GeoTIFF (with SRTM download) and PNG / JPG / unprojected satellite images!

Usage:
    python run_pipeline.py test_area.tif
    python run_pipeline.py city_aerial.png
    python run_pipeline.py test_area.tif --scale 30.0 --radius 60
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

# Fix console encoding on Windows to prevent UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image


# ===================================================================
#  STEP 1 — M1 Inference (Relative Height Model)
# ===================================================================

def run_m1_inference(
    image_path: Path,
    checkpoint: Path,
    model_size: str = "small",
    gsd_m: float = 1.0,
    output_dir: Path = Path("outputs/pipeline"),
) -> Path:
    """Run M1 inference -> save relative height as .npy."""
    import torch
    import torch.nn.functional as F
    from m1.config import M1Config
    from m1.model import HeightPriorNetwork, pick_device

    output_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    print(f"[M1] Loading model (size={model_size}) on {device}")
    cfg = M1Config(size=model_size, freeze_backbone=True)
    net = HeightPriorNetwork(cfg).to(device)

    if checkpoint.exists():
        print(f"[M1] Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ckpt.get("model", ckpt))
    else:
        print(f"[M1] WARNING: No checkpoint at {checkpoint}, using pretrained backbone")

    net.eval()

    pil_img = Image.open(image_path).convert("RGB")
    W, H = pil_img.size
    print(f"[M1] Input: {image_path.name} ({W}x{H})")

    with torch.no_grad():
        out = net.infer_pil(pil_img, gsd_m=gsd_m)
        relative = out.relative[0].cpu().numpy()

    # Resize to match original dimensions if needed
    if relative.shape != (H, W):
        print(f"[M1] Resizing {relative.shape} -> ({H}, {W})")
        pred_t = torch.from_numpy(relative).unsqueeze(0).unsqueeze(0)
        relative = F.interpolate(
            pred_t, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().numpy()

    npy_path = output_dir / f"{image_path.stem}_relative.npy"
    np.save(npy_path, relative.astype(np.float32))
    print(f"[M1] Saved: {npy_path}")
    print(f"[M1]   shape={relative.shape}, range=[{relative.min():.4f}, {relative.max():.4f}]")
    return npy_path


# ===================================================================
#  GeoTIFF & Image helpers
# ===================================================================

def is_valid_latlon_bounds(west: float, south: float, east: float, north: float) -> bool:
    """Validate whether bounds represent real-world latitude/longitude."""
    return (
        -180.0 <= west <= 180.0 and
        -180.0 <= east <= 180.0 and
        -90.0 <= south <= 90.0 and
        -90.0 <= north <= 90.0 and
        south < north and
        west != east
    )


def read_geotiff_geo(img_path: Path) -> dict:
    """Read geo-metadata from a GeoTIFF or standard image (PNG/JPG)."""
    import tifffile

    suffix = img_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            with tifffile.TiffFile(img_path) as tif:
                page = tif.pages[0]
                shape = page.shape[:2]
                tags = {tag.name: tag.value for tag in page.tags.values()}

            pixel_scale = tags.get("ModelPixelScaleTag", (1.0, 1.0, 0.0))
            tiepoint = tags.get("ModelTiepointTag", (0, 0, 0, 0, 0, 0))
            geo_keys = tags.get("GeoKeyDirectoryTag", ())

            epsg = 4326
            if len(geo_keys) >= 16:
                for i in range(4, len(geo_keys), 4):
                    if i + 3 < len(geo_keys) and geo_keys[i] == 2048:
                        epsg = geo_keys[i + 3]

            h, w = shape[0], shape[1]
            origin_x = float(tiepoint[3])
            origin_y = float(tiepoint[4])
            scale_x = float(pixel_scale[0])
            scale_y = float(pixel_scale[1])

            # Check if coordinates are valid lat/lon
            west = origin_x
            north = origin_y
            east = west + w * scale_x
            south = north - h * scale_y
            has_valid_geo = is_valid_latlon_bounds(west, south, east, north)

            return {
                "shape": (h, w),
                "origin_x": origin_x,
                "origin_y": origin_y,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "epsg": epsg,
                "is_geographic": has_valid_geo,
            }
        except Exception as e:
            print(f"[GeoTIFF] Note: Could not read geokeys ({e}), treating as local raster.")

    # Plain PNG, JPG, or unprojected TIFF
    with Image.open(img_path) as img:
        w, h = img.size
    return {
        "shape": (h, w),
        "origin_x": 0.0,
        "origin_y": 0.0,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "epsg": 4326,
        "is_geographic": False,
    }


def write_geotiff(
    data: np.ndarray,
    output_path: Path,
    origin_x: float,
    origin_y: float,
    scale_x: float,
    scale_y: float,
    epsg: int = 4326,
) -> Path:
    """Write a float32 GeoTIFF with proper geo-tags using tifffile."""
    import tifffile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = data.astype(np.float32)

    pixel_scale = (scale_x, scale_y, 0.0)
    tiepoint = (0.0, 0.0, 0.0, origin_x, origin_y, 0.0)

    if epsg == 4326:
        geo_keys = (
            1, 1, 0, 3,
            1024, 0, 1, 2,     # GTModelTypeGeoKey = Geographic
            1025, 0, 1, 1,     # GTRasterTypeGeoKey = PixelIsArea
            2048, 0, 1, epsg,  # GeographicTypeGeoKey = EPSG
        )
    else:
        geo_keys = (
            1, 1, 0, 3,
            1024, 0, 1, 1,     # GTModelTypeGeoKey = Projected
            1025, 0, 1, 1,
            3072, 0, 1, epsg,  # ProjectedCSTypeGeoKey = EPSG
        )

    extra_tags = [
        (33550, 'd', 3, pixel_scale, True),   # ModelPixelScaleTag
        (33922, 'd', 6, tiepoint, True),       # ModelTiepointTag
        (34735, 'H', len(geo_keys), geo_keys, True),  # GeoKeyDirectoryTag
    ]

    tifffile.imwrite(
        str(output_path),
        data,
        dtype=np.float32,
        extratags=extra_tags,
    )
    return output_path


# ===================================================================
#  STEP 2 — Frequency-Split Calibration (v2)
# ===================================================================

def _box_blur(a: np.ndarray, r: int) -> np.ndarray:
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


def lowpass(a: np.ndarray, radius: int, passes: int = 2) -> np.ndarray:
    """Two box passes approximate a Gaussian filter."""
    out = a
    for _ in range(passes):
        out = _box_blur(out, radius)
    return out


def estimate_scale(h: np.ndarray, dem: np.ndarray, radius: int) -> float:
    """
    Estimate scale (metres per unit of h) by regressing the SMOOTH part of h
    against the SMOOTH part of the DEM.
    """
    hl = lowpass(h, radius)
    dl = lowpass(dem, radius)

    m = np.isfinite(hl) & np.isfinite(dl)
    x, y = hl[m], dl[m]

    if x.size < 100 or np.std(x) < 1e-9:
        print("  [Scale] Notice: Not enough variance in smooth component; using default scale 25.0 m/unit.")
        return 25.0

    a = float(np.polyfit(x, y, 1)[0])
    corr = float(np.corrcoef(x, y)[0, 1])
    relief = float(dl.max() - dl.min())

    print(f"  [Scale] Estimated scale : {a:.3f} m per unit of h")
    print(f"  [Scale] Terrain corr    : {corr:+.3f}")
    print(f"  [Scale] DEM relief      : {relief:.1f} m across the scene")

    if abs(relief) < 15:
        print("  [Scale] WARNING: Terrain is nearly flat, so automatic scale is constrained.")
        print("          If buildings look flat, pass --scale (e.g. --scale 30.0).")

    return abs(a) if abs(a) > 0.5 else 25.0


def fetch_srtm_dem(
    west: float, south: float, east: float, north: float,
    out_path: Path,
    api_key: str = "2abedde1f0675abe37066b9daded3e81"
) -> Path:
    """Download SRTM DEM from OpenTopography for the given bounds."""
    import requests

    print(f"[Calibration] Fetching SRTM DEM for bbox: W={west:.4f} S={south:.4f} E={east:.4f} N={north:.4f}")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",
        "south": south, "north": north, "west": west, "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    resp = requests.get(url, params=params, stream=True, timeout=60)
    resp.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[Calibration] SRTM DEM downloaded: {out_path}")
    return out_path


def read_dem_array(dem_path: Path) -> Tuple[np.ndarray, dict]:
    """Read DEM GeoTIFF into numpy array using PIL (handles LZW) + geo metadata via tifffile."""
    with Image.open(dem_path) as img:
        data = np.array(img, dtype=np.float32)
    geo = read_geotiff_geo(dem_path)
    return data, geo


def resample_dem_to_grid(
    dem: np.ndarray,
    dem_geo: dict,
    target_geo: dict,
) -> np.ndarray:
    """Resample DEM to match target grid using bilinear interpolation."""
    from scipy.ndimage import map_coordinates

    tgt_h, tgt_w = target_geo["shape"]

    cols = np.arange(tgt_w)
    rows = np.arange(tgt_h)
    cc, rr = np.meshgrid(cols, rows)

    # Target pixel -> lon/lat
    lon = target_geo["origin_x"] + cc * target_geo["scale_x"]
    lat = target_geo["origin_y"] - rr * target_geo["scale_y"]

    # lon/lat -> DEM pixel coords
    dem_col = (lon - dem_geo["origin_x"]) / dem_geo["scale_x"]
    dem_row = (dem_geo["origin_y"] - lat) / dem_geo["scale_y"]

    # Bilinear interpolation from DEM
    resampled = map_coordinates(
        dem, [dem_row, dem_col], order=1, mode="nearest"
    ).astype(np.float32)

    return resampled


def run_calibration(
    input_tif: Path,
    relative_npy: Path,
    output_tif: Path,
    output_dir: Path = Path("outputs/pipeline"),
    scale: Optional[float] = None,
    radius: Optional[int] = None,
) -> Path:
    """
    Calibrate relative height to metres using frequency-split anchor (v2):
        terrain = lowpass(SRTM)
        detail  = h - lowpass(h)
        dsm     = terrain + scale * detail
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read geo-metadata from input image
    geo = read_geotiff_geo(input_tif)
    print(f"[Calibration] Input: {input_tif.name} ({geo['shape'][1]}x{geo['shape'][0]})")

    # 2. Load relative height
    h = np.load(relative_npy).astype(np.float64)
    if h.shape != geo["shape"]:
        raise ValueError(
            f"Shape mismatch! Image={geo['shape']} but .npy={h.shape}"
        )
    h = np.nan_to_num(h, nan=float(np.nanmedian(h)))
    print(f"[Calibration] Relative height range: [{h.min():.4f}, {h.max():.4f}]")

    # Determine filter radius in pixels (default ~60m cutoff)
    px_m = abs(geo["scale_x"])
    if geo["is_geographic"] and geo["epsg"] == 4326:
        px_m *= 111_320.0  # degrees to approximate metres
    filter_r = radius if radius else max(3, int(round(60.0 / max(px_m, 1e-6))))
    print(f"[Calibration] Pixel resolution: ~{px_m:.2f}m -> Low-pass radius: {filter_r} px (~{filter_r * px_m:.0f}m cutoff)")

    # 3. Check if we can fetch SRTM DEM (geographic bounding box exists)
    if geo["is_geographic"]:
        tgt_h, tgt_w = geo["shape"]
        west = geo["origin_x"]
        north = geo["origin_y"]
        east = west + tgt_w * geo["scale_x"]
        south = north - tgt_h * geo["scale_y"]

        raw_dem_path = output_dir / f"{input_tif.stem}_srtm_tmp.tif"
        fetch_srtm_dem(west, south, east, north, raw_dem_path)

        dem_raw, dem_geo = read_dem_array(raw_dem_path)
        print(f"[Calibration] DEM shape={dem_raw.shape}, resampling to {geo['shape']}...")
        dem = resample_dem_to_grid(dem_raw, dem_geo, geo).astype(np.float64)

        # Clean nodata / sentinel values
        bad = ~np.isfinite(dem) | (dem < -1e4) | (dem > 1e5)
        if bad.any():
            dem[bad] = np.nanmedian(dem[~bad])
            print(f"[Calibration] Filled {int(bad.sum())} nodata cells in DEM.")

        # Determine scale
        if scale is not None:
            actual_scale = scale
            print(f"[Calibration] Using supplied scale: {actual_scale:.3f} m/unit")
        else:
            actual_scale = estimate_scale(h, dem, filter_r)

        # Frequency split: absolute terrain from SRTM + detail from model
        terrain = lowpass(dem, filter_r)
        detail = h - lowpass(h, filter_r)
        dsm = terrain + actual_scale * detail

        print(f"[Calibration] SRTM Range   : {dem.min():.1f} to {dem.max():.1f} m (spread {dem.max() - dem.min():.1f} m)")
        print(f"[Calibration] Calibrated DSM: {dsm.min():.1f} to {dsm.max():.1f} m (spread {dsm.max() - dsm.min():.1f} m)")
        print(f"[Calibration] Building detail added: {float(actual_scale * (detail.max() - detail.min())):.1f} m")

        # Cleanup temp DEM
        if raw_dem_path.exists():
            try:
                os.remove(raw_dem_path)
            except OSError:
                pass

    else:
        # Standard PNG, JPG, or unprojected TIFF without valid lat/lon
        print("[Calibration] Note: Image has no geographic coordinates (PNG/local raster).")
        print("[Calibration] Applying flat-ground + building-detail DSM synthesis.")

        actual_scale = scale if scale is not None else 30.0
        print(f"[Calibration] Using metric scale: {actual_scale:.2f} m per unit")

        # Extract only the high-frequency building/structure detail
        detail = h - lowpass(h, filter_r)
        # Flat ground + sharp building detail — no fake terrain undulation
        dsm = actual_scale * detail
        # Offset so ground level starts around ~5m
        dsm = dsm - dsm.min() + 5.0

        print(f"[Calibration] Synthesized DSM Elevation Range: [{dsm.min():.1f}, {dsm.max():.1f}] m (spread: {dsm.max() - dsm.min():.1f} m)")

    # 4. Write final calibrated DSM GeoTIFF
    output_tif = Path(output_tif)
    write_geotiff(
        dsm.astype(np.float32),
        output_tif,
        origin_x=geo["origin_x"],
        origin_y=geo["origin_y"],
        scale_x=geo["scale_x"],
        scale_y=geo["scale_y"],
        epsg=geo["epsg"],
    )

    print(f"[Calibration] [SUCCESS] Calibrated DSM saved: {output_tif}")
    return output_tif


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DepthWizard: M1 + Frequency-Split Calibration -> 3D DSM GeoTIFF in metres"
    )
    parser.add_argument("input_image", type=Path, help="Input satellite GeoTIFF or PNG/JPG")
    parser.add_argument("--step", choices=["all", "m1", "calibrate"], default="all")
    parser.add_argument("--npy", type=Path, default=None, help="Existing relative .npy (skip M1)")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/m1_small_best.pt"))
    parser.add_argument("--size", default="small", choices=["small", "base", "large"])
    parser.add_argument("--gsd", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=None, help="Metres per unit of h (e.g. 30.0)")
    parser.add_argument("--radius", type=int, default=None, help="Low-pass radius in pixels (default ~60m cutoff)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--output-tif", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_image.stem
    output_tif = args.output_tif or (args.output_dir / f"{stem}_calibrated_dsm.tif")

    print("=" * 60)
    print("  DepthWizard Pipeline: M1 + Calibration (v2 Frequency-Split)")
    print("=" * 60)

    # --- Step 1: M1 ---
    if args.step in ("all", "m1"):
        if args.step == "all" and args.npy is not None and args.npy.exists():
            print(f"[Pipeline] Using provided .npy: {args.npy}")
            npy_path = args.npy
        else:
            npy_path = run_m1_inference(
                image_path=args.input_image,
                checkpoint=args.checkpoint,
                model_size=args.size,
                gsd_m=args.gsd,
                output_dir=args.output_dir,
            )
    elif args.npy is not None:
        npy_path = args.npy
    else:
        npy_path = args.output_dir / f"{stem}_relative.npy"
        if not npy_path.exists():
            npy_path = Path(f"outputs/predictions/{stem}_relative.npy")

    if not npy_path.exists():
        raise FileNotFoundError(f"Relative height file not found: {npy_path}")

    if args.step == "m1":
        print("\n[Done] M1 step completed. To run calibration, execute:")
        print(f"  python run_pipeline.py {args.input_image} --step calibrate --npy {npy_path}")
        return

    # --- Step 2: Calibration ---
    print()
    run_calibration(
        input_tif=args.input_image,
        relative_npy=npy_path,
        output_tif=output_tif,
        output_dir=args.output_dir,
        scale=args.scale,
        radius=args.radius,
    )

    print()
    print("=" * 60)
    print(f"  [SUCCESS] Finished! Calibrated GeoTIFF: {output_tif}")
    print("=" * 60)


if __name__ == "__main__":
    main()
