import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class GlendaDataset(Dataset):
    def __init__(self, split_dir, transform=None):
        """
        split_dir: 'train', 'valid' or 'test'
        """
        self.split_dir = split_dir
        self.transform = transform
        
        self.mask_filenames = [f for f in os.listdir(split_dir) if f.startswith('mask_')]

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask_name = self.mask_filenames[idx]
        
        img_name = mask_name.replace("mask_", "").replace(".png", ".jpg")
        
        img_path = os.path.join(self.split_dir, img_name)
        mask_path = os.path.join(self.split_dir, mask_name)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        mask = mask.unsqueeze(0)

        return image, mask