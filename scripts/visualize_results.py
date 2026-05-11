import os
import sys
import torch
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.models import get_model
from src.utils import get_device
from src.transforms import get_val_transforms
from src.visualization import plot_learning_curves, generate_split_outputs

def run_visualization(model=None, device=None, history=None):
    if device is None:
        device = get_device()
    
    if model is None:
        print("Loading model...")
        model = get_model('unet').to(device)
        if os.path.exists(config.BASELINE_WEIGHTS):
            model.load_state_dict(torch.load(config.BASELINE_WEIGHTS, map_location=device))
        else:
            print(f"ERROR: not found path {config.BASELINE_WEIGHTS}")
            return
    
    model.eval()

    print("Generating overlay for test images")
    test_results_path = os.path.join(config.OUTPUT_DIR, "test_results")
    
    generate_split_outputs(
        model, 
        device, 
        os.path.join(config.DATA_DIR, "test"), 
        test_results_path,
        max_images=None # process everything
    )
    
    print("Generated overlay for test images")

    if history is None:
        history_path = os.path.join(config.OUTPUT_DIR, "history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
    
    if history:
        plot_learning_curves(
            history, 
            save_path=os.path.join(config.OUTPUT_DIR, "learning_curves.png")
        )
    else:
        print("Not history found to be plotted")

if __name__ == "__main__":
    # 'python scripts/visualize_results.py'
    run_visualization()