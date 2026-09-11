"""
run_pipeline.py — Club M1 (relative height model) + Mohana's DEM calibration.
Uses tifffile + PIL + scipy instead of rasterio (bypassing Windows Application Control DLL block).

Usage:
    # 1. Full pipeline in one shot:
    python run_pipeline.py test_area.tif

    # 2. Step 1 only — Run M1 model to produce relative height .npy:
    python run_pipeline.py test_area.tif --step m1

    # 3. Step 2 only — Feed an existing relative .npy into calibration:
    python run_pipeline.py test_area.tif --step calibrate --npy outputs/predictions/test_area_relative.npy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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

    # Resize to match original if needed
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
#  GeoTIFF helpers (using tifffile + PIL, NOT rasterio)
# ===================================================================

def read_geotiff_geo(tif_path: Path) -> dict:
    """Read geo-metadata from a GeoTIFF using tifffile."""
    import tifffile

    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        shape = page.shape  # (H, W, ...) or (H, W)
        tags = {tag.name: tag.value for tag in page.tags.values()}

    pixel_scale = tags.get("ModelPixelScaleTag", (1.0, 1.0, 0.0))
    tiepoint = tags.get("ModelTiepointTag", (0, 0, 0, 0, 0, 0))
    geo_keys = tags.get("GeoKeyDirectoryTag", ())

    epsg = 4326  # default
    if len(geo_keys) >= 16:
        for i in range(4, len(geo_keys), 4):
            if i + 3 < len(geo_keys) and geo_keys[i] == 2048:
                epsg = geo_keys[i + 3]

    h = shape[0]
    w = shape[1] if len(shape) > 1 else 1

    origin_x = tiepoint[3]  # longitude (or easting)
    origin_y = tiepoint[4]  # latitude (or northing)
    scale_x = pixel_scale[0]
    scale_y = pixel_scale[1]

    return {
        "shape": (h, w),
        "origin_x": origin_x,
        "origin_y": origin_y,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "epsg": epsg,
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
#  STEP 2 — Calibration (Mohana's logic, rasterio-free)
# ===================================================================

def fetch_srtm_dem(west, south, east, north, out_path, api_key="2abedde1f0675abe37066b9daded3e81"):
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


def read_dem_array(dem_path: Path) -> tuple[np.ndarray, dict]:
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
) -> Path:
    """Calibrate relative height to metres using SRTM DEM, write GeoTIFF."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read geo-metadata from input GeoTIFF
    geo = read_geotiff_geo(input_tif)
    print(f"[Calibration] Input GeoTIFF: {input_tif}")
    print(f"[Calibration]   shape={geo['shape']}, EPSG:{geo['epsg']}")
    print(f"[Calibration]   origin=({geo['origin_x']:.6f}, {geo['origin_y']:.6f})")
    print(f"[Calibration]   pixel scale=({geo['scale_x']:.8f}, {geo['scale_y']:.8f})")

    # 2. Load relative height
    h = np.load(relative_npy)
    if h.shape != geo["shape"]:
        raise ValueError(
            f"Shape mismatch! GeoTIFF={geo['shape']} but .npy={h.shape}"
        )
    print(f"[Calibration] Relative height: shape={h.shape}, range=[{h.min():.4f}, {h.max():.4f}]")

    # 3. Compute lat/lon bounds
    tgt_h, tgt_w = geo["shape"]
    west = geo["origin_x"]
    north = geo["origin_y"]
    east = west + tgt_w * geo["scale_x"]
    south = north - tgt_h * geo["scale_y"]

    # 4. Fetch SRTM DEM
    raw_dem_path = output_dir / f"{input_tif.stem}_srtm_tmp.tif"
    fetch_srtm_dem(west, south, east, north, raw_dem_path)

    # 5. Read and resample DEM to our pixel grid
    dem_raw, dem_geo = read_dem_array(raw_dem_path)
    print(f"[Calibration] DEM shape={dem_raw.shape}, resampling to {geo['shape']}...")
    dem_resampled = resample_dem_to_grid(dem_raw, dem_geo, geo)

    # 6. Fit affine: real_elevation = a * h + b (ignoring SRTM nodata < -100)
    valid = np.isfinite(h) & np.isfinite(dem_resampled) & (dem_resampled > -100)
    x = h[valid].astype(np.float64)
    y = dem_resampled[valid].astype(np.float64)

    if x.size < 8:
        raise ValueError(f"Only {x.size} valid pixels — need at least 8 to fit")

    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    fit_rmse = np.sqrt(np.mean((pred - y) ** 2))
    print(f"[Calibration] Fitted transform: real_elevation = {a:.4f} * h + {b:.4f}")
    print(f"[Calibration] Fit RMSE: {fit_rmse:.2f} m")

    # 7. Apply transform -> real elevation in metres
    real_elevation = (a * h + b).astype(np.float32)
    print(f"[Calibration] Elevation range: [{real_elevation.min():.1f}, {real_elevation.max():.1f}] m")

    # 8. Write calibrated DSM GeoTIFF
    output_tif = Path(output_tif)
    write_geotiff(
        real_elevation, output_tif,
        origin_x=geo["origin_x"],
        origin_y=geo["origin_y"],
        scale_x=geo["scale_x"],
        scale_y=geo["scale_y"],
        epsg=geo["epsg"],
    )

    # Cleanup temp DEM
    if raw_dem_path.exists():
        try:
            os.remove(raw_dem_path)
        except OSError:
            pass

    print(f"[Calibration] [SUCCESS] Calibrated DSM saved: {output_tif}")
    return output_tif


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DepthWizard: M1 + Calibration -> GeoTIFF in metres"
    )
    parser.add_argument("input_tif", type=Path, help="Input satellite GeoTIFF")
    parser.add_argument("--step", choices=["all", "m1", "calibrate"], default="all")
    parser.add_argument("--npy", type=Path, default=None, help="Existing relative .npy (skip M1)")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/m1_small_best.pt"))
    parser.add_argument("--size", default="small", choices=["small", "base", "large"])
    parser.add_argument("--gsd", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pipeline"))
    parser.add_argument("--output-tif", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_tif.stem
    output_tif = args.output_tif or (args.output_dir / f"{stem}_calibrated_dsm.tif")

    print("=" * 60)
    print("  DepthWizard Pipeline: M1 + Calibration -> GeoTIFF")
    print("=" * 60)

    # --- Step 1: M1 ---
    if args.step in ("all", "m1"):
        if args.step == "all" and args.npy is not None and args.npy.exists():
            print(f"[Pipeline] Using provided .npy: {args.npy}")
            npy_path = args.npy
        else:
            npy_path = run_m1_inference(
                image_path=args.input_tif,
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
        print(f"  python run_pipeline.py {args.input_tif} --step calibrate --npy {npy_path}")
        return

    # --- Step 2: Calibration ---
    print()
    run_calibration(
        input_tif=args.input_tif,
        relative_npy=npy_path,
        output_tif=output_tif,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 60)
    print(f"  [SUCCESS] Finished! Calibrated GeoTIFF: {output_tif}")
    print("=" * 60)


if __name__ == "__main__":
    main()
