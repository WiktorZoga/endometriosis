import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

SEED = 2137

DATA_DIR = os.path.join(ROOT_DIR, "Glenda-1") 
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

IMAGE_SIZE = (256, 256)
IN_CHANNELS = 3
NUM_CLASSES = 1

THRESHOLD = 0.5

# ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BASELINE_WEIGHTS = os.path.join(OUTPUT_DIR, "best_baseline.pth")
SWIN_WEIGHTS = os.path.join(OUTPUT_DIR, "best_swin.pth")