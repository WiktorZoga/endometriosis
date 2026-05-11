import torch
from tqdm import tqdm
from src.utils import calculate_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            
            dice, iou = calculate_metrics(outputs, masks)
            val_dice += dice.item()
            val_iou += iou.item()
            
    return val_loss / len(dataloader), val_dice / len(dataloader), val_iou / len(dataloader)