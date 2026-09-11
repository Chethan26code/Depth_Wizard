import json

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation
)

INPUT = "data/processed/rgb.png"
OUTPUT = 'data/processed/relative_depth.npy'
OUTPUT_JSON = 'data/processed/metadata.json'

MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"

#SLEECT DEVICE

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("Using device:", device)

#LOAD MODEL

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

print("Model loaded")

#LOAD IMAGE
image = Image.open(INPUT).convert("RGB")
width, height = image.size
print("Image size:",width,"x",height)

