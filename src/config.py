import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import random
import numpy as np

SEED = 2137
DATA_DIR = "Glenda-1" 
OUTPUT_DIR = "outputs"

IMAGE_SIZE = (256, 256)
IN_CHANNELS = 3
NUM_CLASSES = 1

THRESHOLD = 0.5

# ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BASELINE_WEIGHTS = "best_baseline.pth"
SWIN_WEIGHTS = "best_swin.pth"

def get_train_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_seed(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed
