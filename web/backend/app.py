"""
DepthWizard Web Application — Backend Server
Location: web/backend/app.py

Connects the M1 Deep Learning Model + Mohana's DEM Calibration
with Maansi's Babylon.js 3D DSM Web Viewer (web/frontend/index.html).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Safe UTF-8 encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Define directory layout
BACKEND_DIR = Path(__file__).resolve().parent
WEB_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEB_DIR.parent
FRONTEND_DIR = WEB_DIR / "frontend"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pipeline"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to sys.path so modules like run_pipeline and m1 are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_file, send_from_directory
from PIL import Image
import numpy as np

# Import unified pipeline functions
from run_pipeline import (
    run_m1_inference,
    run_calibration,
    read_geotiff_geo,
)

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="",
)


def export_rgb_png(input_path: Path, output_png_path: Path) -> Path:
    """Converts a satellite GeoTIFF or image to a standard RGB PNG for 3D texturing."""
    with Image.open(input_path) as img:
        rgb = img.convert("RGB")
        rgb.save(output_png_path, "PNG")
    return output_png_path


@app.route("/")
def index():
    """Serve the 3D DSM frontend viewer."""
    return send_file(FRONTEND_DIR / "index.html")


@app.route("/api/samples", methods=["GET"])
def list_samples():
    """List sample satellite imagery available in the workspace."""
    samples = []
    for p in PROJECT_ROOT.glob("*.tif"):
        samples.append({"name": p.name, "path": p.name, "size_bytes": p.stat().st_size})
    for p in (PROJECT_ROOT / "data").glob("**/*.tif"):
        samples.append({"name": p.name, "path": str(p.relative_to(PROJECT_ROOT)), "size_bytes": p.stat().st_size})
    return jsonify({"samples": samples})


@app.route("/api/process", methods=["POST"])
def process_pipeline():
    """
    Accepts an uploaded image or sample name, runs M1 + DEM Calibration,
    and returns URLs for the calibrated 3D DSM GeoTIFF and RGB texture.
    """
    try:
        input_file_path: Optional[Path] = None

        # 1. Check for uploaded file
        if "file" in request.files and request.files["file"].filename:
            uploaded_file = request.files["file"]
            filename = uploaded_file.filename
            save_path = UPLOAD_DIR / filename
            uploaded_file.save(save_path)
            input_file_path = save_path
        else:
            # Handle sample selection
            data = request.form if request.form else (request.get_json(silent=True) or {})
            sample_name = data.get("sample")
            if sample_name:
                cand = PROJECT_ROOT / sample_name
                if cand.exists():
                    input_file_path = cand

        if not input_file_path or not input_file_path.exists():
            default_cand = PROJECT_ROOT / "test_area.tif"
            if default_cand.exists():
                input_file_path = default_cand
            else:
                return jsonify({"error": "No input image provided or found."}), 400

        stem = input_file_path.stem
        output_tif = OUTPUT_DIR / f"{stem}_calibrated_dsm.tif"
        output_npy = OUTPUT_DIR / f"{stem}_relative.npy"
        output_png = OUTPUT_DIR / f"{stem}_rgb.png"

        print(f"[API] Processing {input_file_path.name} -> DSM GeoTIFF")

        # 2. Export RGB texture for Babylon.js draping
        export_rgb_png(input_file_path, output_png)

        # 3. Run M1 Inference (or reuse if requested)
        reuse_npy = request.form.get("reuse_npy", "false").lower() == "true"
        if not (reuse_npy and output_npy.exists()):
            run_m1_inference(
                image_path=input_file_path,
                checkpoint=PROJECT_ROOT / "outputs" / "checkpoints" / "m1_small_best.pt",
                model_size="small",
                gsd_m=1.0,
                output_dir=OUTPUT_DIR,
            )

        # 4. Run DEM Calibration -> Produces Calibrated DSM GeoTIFF (metres)
        run_calibration(
            input_tif=input_file_path,
            relative_npy=output_npy,
            output_tif=output_tif,
            output_dir=OUTPUT_DIR,
        )

        # 5. Read metadata for UI
        geo_info = read_geotiff_geo(output_tif)
        relative_data = np.load(output_npy)

        return jsonify({
            "status": "success",
            "filename": input_file_path.name,
            "dsm_url": f"/outputs/pipeline/{output_tif.name}",
            "rgb_url": f"/outputs/pipeline/{output_png.name}",
            "npy_url": f"/outputs/pipeline/{output_npy.name}",
            "meta": {
                "shape": geo_info["shape"],
                "origin": [geo_info["origin_x"], geo_info["origin_y"]],
                "scale": [geo_info["scale_x"], geo_info["scale_y"]],
                "epsg": geo_info["epsg"],
                "relative_range": [float(relative_data.min()), float(relative_data.max())],
            }
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/outputs/pipeline/<path:filename>")
def serve_output(filename):
    """Serve output pipeline artifacts (GeoTIFF, .npy, PNG textures)."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  DepthWizard 3D DSM Web Application")
    print(f"  Running on: http://127.0.0.1:{port}")
    print(f"  Frontend: {FRONTEND_DIR / 'index.html'}")
    print(f"  Backend:  {BACKEND_DIR / 'app.py'}")
    print("=" * 60)
    app.run(host="127.0.0.1", port=port, debug=False)
