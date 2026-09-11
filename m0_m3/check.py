import rasterio
import json

with rasterio.open("/Users/meenakshik/Downloads/test_area_dsm.tif") as src:
    print(f"Shape: {src.width} x {src.height}")
    print(f"CRS: {src.crs}")
    print(f"dtype: {src.dtypes}")
    print(f"Band count: {src.count}")

with open("data/tiles/test_area_tif/manifest.json") as f:
    manifest = json.load(f)
print(f"\nExpected tile size: {manifest['tile_size']} x {manifest['tile_size']}")
print(f"Expected full image size: {manifest['tiling_info']['original_width']} x {manifest['tiling_info']['original_height']}")