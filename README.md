# DepthWizard — Satellite to 3D Calibrated DSM Pipeline & Web Viewer

This repository integrates the **M1 Relative Height Prior** (Depth Anything V2 + Height Head) with **DEM Anchor Calibration** and a **Babylon.js 3D Terrain Viewer** to convert raw satellite imagery into interactive 3D digital surface models (with real elevation in metres).

---

## 🚀 Interactive 3D Web Application

Run the integrated server:
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser:
- **Select or Upload** any satellite image / GeoTIFF.
- Click **"⚡ Run ML Model & Build 3D DSM"**:
  1. Runs M1 (Depth Anything V2) to extract relative height.
  2. Fetches SRTM 30m DEM and calibrates elevation to metres.
  3. Drapes the satellite image texture onto the 3D elevation mesh in Babylon.js.
- **Interactive Tools**:
  - Orbit camera & First-Person Fly (WASD) navigation.
  - Real-time Sun position & directional shadow simulation.
  - Click any rooftop or terrain point for true height, slope, and distance measurements.
  - Download calibrated DSM GeoTIFF (`.tif`) and raw relative height (`.npy`).

---

## Command-Line Pipeline Execution

The script [`run_pipeline.py`](run_pipeline.py) connects M1 inference and DEM calibration in a single, rasterio-free workflow (immune to Windows Application Control / DLL blocking).

### Option 1: End-to-End in One Shot
Takes input satellite GeoTIFF -> runs M1 -> fetches SRTM DEM -> fits anchor transform -> outputs calibrated GeoTIFF in metres:
```bash
python run_pipeline.py test_area.tif
```
- **Output GeoTIFF**: `outputs/pipeline/test_area_calibrated_dsm.tif`
- **Output Relative .npy**: `outputs/pipeline/test_area_relative.npy`

---

### Option 2: Step-by-Step (Manual Inspection of .npy)

#### Step A: Run M1 Model to Generate Relative Height `.npy`
```bash
python run_pipeline.py test_area.tif --step m1
```
This generates:
- `outputs/pipeline/test_area_relative.npy`

#### Step B: Feed that `.npy` Manually into Calibration
Point to your generated (or manually located) `.npy` array:
```bash
python run_pipeline.py test_area.tif --step calibrate --npy outputs/pipeline/test_area_relative.npy
```
This performs:
1. Geo-referencing & bbox extraction (`EPSG:4326`)
2. SRTM 30m DEM fetching via OpenTopography API
3. Bilinear grid resampling
4. Linear regression fitting (`real_elevation = a * h + b`)
5. Saving the final calibrated DSM GeoTIFF to `outputs/pipeline/test_area_calibrated_dsm.tif`

---

## Original M1 Modules (Standalone)

```bash
python -m m1.lesson0_scale                 # Scale geometry
python -m m1.zeroshot                      # Zero-shot baseline evaluation
python -m m1.train --epochs 1 --tiles 8    # Synthetic tile training
python -m m1.inference --image test_area.tif --out outputs/predictions/test_area_prediction.png
```
