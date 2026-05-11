import segmentation_models_pytorch as smp
from monai.networks.nets import SwinUNETR
from src import config

def get_unet():
    """Baseline Unet"""
    model = smp.Unet(
        encoder_name="resnet34",        
        encoder_weights="imagenet",     
        in_channels=config.IN_CHANNELS,                  
        classes=config.NUM_CLASSES,                      
    )
    return model

def get_swin_unetr():
    """Swin-Unet"""
    model = SwinUNETR(
        img_size=config.IMAGE_SIZE,
        in_channels=config.IN_CHANNELS,
        out_channels=config.NUM_CLASSES,
        feature_size=24,
        spatial_dims=2 
    )
    return model

def get_model(model_name):
    if model_name.lower() == 'unet':
        return get_unet()
    elif model_name.lower() == 'swin':
        return get_swin_unetr()
    else:
        raise ValueError(f"Unknown model: {model_name}. Select 'unet' or 'swin'.")