import rasterio

INPUT = "data/raw/test_area.tif"

with rasterio.open(INPUT) as src:

    print("======== IMAGE INFORMATION ========")
    print("Width:", src.width)
    print("Height:", src.height)
    print("Bands:", src.count)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Transform:", src.transform)
    print("Data types:", src.dtypes)

    print()
    print("Band Descriptions:")
    for i, description in enumerate(src.descriptions, start=1):
        print(f"Band {i}: {description}")