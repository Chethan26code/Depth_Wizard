import rasterio
import numpy as np
from PIL import Image

INPUT = "data/raw/test_area.tif"
OUTPUT = "data/processed/rgb.png"

with rasterio.open(INPUT) as src:
    #reading firsyt 3 abnds
    rgb = src.read([1,2,3])

    #by default rasterio gives (bands,heifht,width)
    #but we wnat (height, width,bamds) so we transpose the matric

    rgb = np.transpose(rgb, (1,2,0))

    rgb = np.clip(rgb,0,255) #kinda dagerous rn but will fix later

    rgb = rgb.astype(np.uint8)

    Image.fromarray(rgb).save(OUTPUT)

print("Saved:",OUTPUT)
print("Shape:",rgb.shape)