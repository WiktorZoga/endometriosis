import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

from src import config

def generate_split_outputs(model, device, split_dir, output_dir, max_images=None):
    """
    Przechodzi przez zdjęcia w folderze, generuje predykcję i zapisuje siatkę 4 obrazów.
    Oryginał | GT | Predykcja | Nałożenie Przeźroczyste
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Pobieramy same zdjęcia (pliki .jpg), ignorując pliki masek (zaczynające się od 'mask_')
    image_files = [f for f in os.listdir(split_dir) if f.endswith('.jpg') and not f.startswith('mask_')]
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    transform = config.get_val_transforms()

    print(f"Generowanie wyników dla: {split_dir} -> {output_dir}")
    
    for img_name in tqdm(image_files):
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
            pred_mask = (torch.sigmoid(output) > config.THRESHOLD).float().cpu().squeeze().numpy()
            
        img_show = cv2.resize(image_rgb, config.IMAGE_SIZE)
        
        # Przygotowanie obrazu dla 4 komórki (Nałożenie Przeźroczyste)
        # Zaczynamy od oryginału
        img_overlay = img_show.copy()
        
        # 4. Sprawdzanie czy Ground Truth (Maska lekarza) istnieje
        has_gt = os.path.exists(mask_path)
        
        if has_gt:
            # Tworzymy wykres z 4 komórkami
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Wczytujemy GT
            true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            true_mask = cv2.resize(true_mask, config.IMAGE_SIZE)
            
            # --- Przygotowanie 4 komórki (Overlay) ---
            # 1. Nakładanie GT (Zielony)
            # Tworzymy maskę kolorową [same zielone piksele tam, gdzie GT==255]
            # Używamy formatu RGB: [Red, Green, Blue]
            color_gt_mask = np.zeros_like(img_overlay)
            color_gt_mask[true_mask == 255] = [0, 255, 0] # [R, G, B] - Zielony

            # Nakładanie przeźroczyste (alfa=0.7)
            # Gwarantujemy, że true_mask i true_mask mają ten sam kształt przez dodanie wymiaru kanałów,
            # lub przez zastosowanie powyższego triku z np.zeros_like, co jest bezpieczniejsze.
            img_overlay = cv2.addWeighted(img_overlay, 1.0, color_gt_mask, 0.7, 0)

            # 2. Nakładanie Predykcji (Niebieski)
            # Predykcja jest float (0.0 lub 1.0). Konwersja na uint8 do kolorowania
            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)

            # Tworzymy maskę kolorową [same niebieskie piksele tam, gdzie pred==255]
            color_pred_mask = np.zeros_like(img_overlay)
            color_pred_mask[pred_mask_uint8 == 255] = [0, 0, 255] # [R, G, B] - Niebieski

            # Nakładanie przeźroczyste (alfa=0.7)
            img_overlay = cv2.addWeighted(img_overlay, 1.0, color_pred_mask, 0.7, 0)
            
            axes[0].imshow(img_show)
            axes[0].set_title("Oryginał")
            axes[1].imshow(true_mask, cmap='gray')
            axes[1].set_title("Ground Truth (Lekarz)")
            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title("Predykcja modelu")
            
            axes[3].imshow(img_overlay)
            axes[3].set_title("Nałożenie (Ziel=GT, Nieb=Pred)")

        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            pred_mask_uint8 = (pred_mask * 255).astype(np.uint8)
            color_pred_mask = np.zeros_like(img_overlay)
            color_pred_mask[pred_mask_uint8 == 255] = [0, 0, 255] # Niebieski
            img_overlay = cv2.addWeighted(img_overlay, 1.0, color_pred_mask, 0.7, 0)
            
            axes[0].imshow(img_show)
            axes[0].set_title("Oryginał (Brak GT)")
            axes[1].imshow(pred_mask, cmap='gray')
            axes[1].set_title("Predykcja U-Net")
            axes[2].imshow(img_overlay)
            axes[2].set_title("Nałożenie Predykcji (Nieb)")
            
        for ax in axes:
            ax.axis('off')
            
        save_path = os.path.join(output_dir, f"res_overlay_{img_name}")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) 

if __name__ == "__main__":
    seed = config.set_seed()
    device = config.get_device()
    print(f"Start generowania predykcji... Device: {device}")
    
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights=None,     
        in_channels=3,                  
        classes=1,                      
    )
    
    model_weights = config.BASELINE_WEIGHTS
    if os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        generate_split_outputs(model, device, os.path.join(config.DATA_DIR, "train"), "outputs/train_overlay", max_images=None)
        generate_split_outputs(model, device, os.path.join(config.DATA_DIR, "valid"), "outputs/valid_overlay", max_images=None)
        generate_split_outputs(model, device, os.path.join(config.DATA_DIR, "test"), "outputs/test_overlay", max_images=None) 
        
        print("Gotowe! Zajrzyj do nowych folderów '*_overlay/'.")
    else:
        print(f"Błąd: Nie znaleziono pliku wag {model_weights}. Najpierw wytrenuj model!")