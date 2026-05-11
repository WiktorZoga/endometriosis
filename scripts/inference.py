import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.utils import get_device, set_seed
from src.models import get_model
from src.transforms import get_val_transforms

def predict_single_image(image_path, model_name='unet', weights_path=None, output_path=None):
    """
    Perform segmentation on a single input image using the trained model.
    """

    device = get_device()
    set_seed(config.SEED)
    print(f"[*] Running inference on device: {device}")

    print(f"[*] Initializing {model_name.upper()} architecture...")
    model = get_model(model_name).to(device)
    
    if weights_path is None:
        weights_path = config.BASELINE_WEIGHTS if model_name == 'unet' else config.SWIN_WEIGHTS
        
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[-] Model weights not found at: {weights_path}")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[-] Input image not found at: {image_path}")

    print(f"[*] Loading image: {image_path}")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image_rgb.shape[:2]

    transform = get_val_transforms()
    transformed = transform(image=image_rgb)
    tensor_image = transformed["image"].unsqueeze(0).to(device)

    print("[*] Running prediction...")
    with torch.no_grad():
        output = model(tensor_image)
        pred_mask = (torch.sigmoid(output) > config.THRESHOLD).float().cpu().squeeze().numpy()

    pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    if output_path:
        mask_to_save = (pred_mask_resized * 255).astype(np.uint8)
        cv2.imwrite(output_path, mask_to_save)
        print(f"[+] Prediction mask saved to: {output_path}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(pred_mask_resized, cmap='gray')
        axes[1].set_title(f"{model_name.upper()} Prediction")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for single image segmentation")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image file")
    parser.add_argument("--model", type=str, default='unet', help="Model architecture (unet/swin)")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (optional)")
    parser.add_argument("--output", type=str, default=None, help="Path to save the result mask (optional)")
    
    args = parser.parse_args()
    
    predict_single_image(
        image_path=args.image, 
        model_name=args.model, 
        weights_path=args.weights, 
        output_path=args.output
    )