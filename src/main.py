import os
import cv2
import torch
import argparse
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from src import config

def predict_single_image(image_path, model_path, output_path=None):
    device = config.get_device()
    seed = config.set_seed()
    
    print(f"Używane urządzenie: {device}")

    print("Ładowanie modelu U-Net...")
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None, 
        in_channels=3,                  
        classes=1,                      
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono pliku wag: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Nie znaleziono obrazka: {image_path}")

    print(f"Przetwarzanie obrazu: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_size = image.shape[:2]

    transform = config.get_val_transforms()

    transformed = transform(image=image)
    tensor_image = transformed["image"].unsqueeze(0).to(device) # [1, C, H, W]

    print("Wykonywanie predykcji...")
    with torch.no_grad():
        output = model(tensor_image)
        pred_mask = (torch.sigmoid(output) > config.THRESHOLD).float().cpu().squeeze().numpy()

    pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    if output_path:
        mask_to_save = (pred_mask_resized * 255).astype(np.uint8)
        cv2.imwrite(output_path, mask_to_save)
        print(f"Maskę zapisano w: {output_path}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image)
        axes[0].set_title("Oryginał")
        axes[0].axis("off")
        
        axes[1].imshow(pred_mask_resized, cmap='gray')
        axes[1].set_title("Predykcja Sieci")
        axes[1].axis("off")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testowanie modelu na pojedynczym zdjęciu")
    parser.add_argument("--image", type=str, required=True, help="Ścieżka do zdjęcia wejściowego")
    parser.add_argument("--model", type=str, default=config.BASELINE_WEIGHTS, help="Ścieżka do wag modelu")
    parser.add_argument("--output", type=str, default=None, help="Ścieżka zapisu (jeśli brak, wyświetli obrazek)")
    
    args = parser.parse_args()
    
    predict_single_image(args.image, args.model, args.output)