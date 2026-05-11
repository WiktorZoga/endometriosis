import os
import sys
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.transforms import get_val_transforms
from src.dataset import GlendaDataset
from src.models import get_model
from src.engine import evaluate
from src.utils import get_device

def main():
    device = get_device()
    print(f"Running evaluation on: {device}")

    test_ds = GlendaDataset(
        os.path.join(config.DATA_DIR, "test"), 
        transform=get_val_transforms()
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    print(f"Loaded {len(test_ds)} test images.")

    model_name = 'unet' 
    model = get_model(model_name).to(device)

    weights_path = config.BASELINE_WEIGHTS if model_name == 'unet' else config.SWIN_WEIGHTS
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from: {weights_path}")
    else:
        print(f"ERROR: weights not found at {weights_path}")
        return

    criterion = smp.losses.DiceLoss(mode='binary')

    print(f"Testing model {model_name.upper()}...")
    test_loss, test_dice, test_iou = evaluate(model, test_loader, criterion, device)

    print("\n" + "="*40)
    print("FINAL RESULTS ON TEST DATASET")
    print("="*40)
    print(f"Model: {model_name.upper()}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Dice Score: {test_dice:.4f}")
    print(f"Test IoU Score:  {test_iou:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()