import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config
from src.transforms import get_val_transforms

def generate_split_outputs(model, device, split_dir, output_dir, max_images=None):
    """
    Cycles through the photos in the folder, generates a prediction, and saves a grid of 4 images.
    Original | GT (Doctor) | Prediction | Mask on image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(split_dir) if f.endswith('.jpg') and not f.startswith('mask_')]
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    transform = get_val_transforms()

    print(f"Generating results for: {split_dir} -> {output_dir}")
    
    for img_name in tqdm(image_files, desc="Processing images", leave=False):
        img_path = os.path.join(split_dir, img_name)
        mask_name = "mask_" + img_name.replace(".jpg", ".png")
        mask_path = os.path.join(split_dir, mask_name)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image_rgb)
        tensor_image = transformed["image"].unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor_image)
            pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().squeeze().numpy()
            
        img_show = cv2.resize(image_rgb, config.IMAGE_SIZE)
        img_overlay = img_show.copy()
        
        has_gt = os.path.exists(mask_path)
        
        if has_gt:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.resize(true_mask, config.IMAGE_SIZE)
            
            color_gt_mask = np.zeros_like(img_overlay)
            color_gt_mask[true_mask == 255] = [0, 255, 0] # GREEN for Doctor
            img_overlay = cv2.addWeighted(img_overlay, 1.0, color_gt_mask, 0.7, 0)

            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
            color_pred_mask = np.zeros_like(img_overlay)
            color_pred_mask[pred_mask_uint8 == 255] = [0, 0, 255] # BLUE for Model
            img_overlay = cv2.addWeighted(img_overlay, 1.0, color_pred_mask, 0.7, 0)
            
            axes[0].imshow(img_show)
            axes[0].set_title("Original")
            axes[1].imshow(true_mask, cmap='gray')
            axes[1].set_title("Ground Truth (Doctor)")
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title("Model prediction")
            axes[3].imshow(img_overlay)
            axes[3].set_title("Masks (GT=green, Model=blue)")

        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
            color_pred_mask = np.zeros_like(img_overlay)
            color_pred_mask[pred_mask_uint8 == 255] = [0, 0, 255] # BLUE
            img_overlay = cv2.addWeighted(img_overlay, 1.0, color_pred_mask, 0.7, 0)
            
            axes[0].imshow(img_show)
            axes[0].set_title("Original (no GT)")
            axes[1].imshow(pred_mask, cmap='gray')
            axes[1].set_title("Model prediction")
            axes[2].imshow(img_overlay)
            axes[2].set_title("Model masks (blue)")
            
        for ax in axes:
            ax.axis('off')
            
        save_path = os.path.join(output_dir, f"res_overlay_{img_name}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) 

def plot_learning_curves(history, save_path=None):
    """Plot Loss, Dice i IoU."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0].set_title('Loss History')
    axes[0].set_xlabel('Epochs')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(epochs, history["val_dice"], 'g-', label='Val Dice')
    axes[1].plot(epochs, history["val_iou"], 'm-', label='Val IoU')
    axes[1].set_title('Metrics History')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()