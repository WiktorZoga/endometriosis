import albumentations as A
from albumentations.pytorch import ToTensorV2
from src import config

def get_train_transforms():
    """Transforms with augmentation"""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    """Transforms with augmentation without augmentation"""
    return A.Compose([
        A.Resize(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]),
        A.Normalize(mean=config.MEAN, std=config.STD),
        ToTensorV2(),
    ])