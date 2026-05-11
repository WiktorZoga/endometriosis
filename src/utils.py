import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

def set_seed(seed=2137):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def calculate_metrics(pred, target):
    """Dice and IoU scores"""
    pred = (torch.sigmoid(pred) > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-8)
    iou = intersection / (union + 1e-8)
    
    return dice, iou