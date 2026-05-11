import os
import sys
import torch
import json
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.utils import set_seed, get_device
from src.transforms import get_train_transforms, get_val_transforms
from src.dataset import GlendaDataset
from src.models import get_model
from src.engine import train_one_epoch, evaluate

def main():
    set_seed(config.SEED)
    device = get_device()
    print(f"Starting training on: {device}")

    train_ds = GlendaDataset(
        os.path.join(config.DATA_DIR, "train"), 
        transform=get_train_transforms()
    )
    val_ds = GlendaDataset(
        os.path.join(config.DATA_DIR, "valid"), 
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    print(f"Data loaded: Train ({len(train_ds)}), Validation ({len(val_ds)})")

    model_name = 'unet' # 'swin' przy kolejnym eksperymencie
    model = get_model(model_name).to(device)

    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 20
    patience = 5  # Early stopping
    epochs_no_improve = 0
    best_val_dice = 0.0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": []
    }

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: T={train_loss:.4f}, V={val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            
            save_path = config.BASELINE_WEIGHTS if model_name == 'unet' else config.SWIN_WEIGHTS
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at: {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print(f"Training stopped after {patience} epochs of no improvement.")
            break

    history_save_path = os.path.join(config.OUTPUT_DIR, "history.json")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(history_save_path, "w") as f:
        json.dump(history, f, indent=4)
    
    print(f"Training hisotry saved at: {history_save_path}")
    print(f"Training finished - Best Validation Dice Score: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()