import os
import json
import numpy as np
import cv2
from dotenv import load_dotenv
from roboflow import Roboflow

def convert_coco_to_masks(split_folder):
    """Converts _annotations.coco.json to binary masks PNG."""
    json_path = os.path.join(split_folder, "_annotations.coco.json")
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}

    from collections import defaultdict
    annotations_by_image = defaultdict(list)
    for ann in coco_data.get('annotations', []): 
        annotations_by_image[ann['image_id']].append(ann)

    print(f"Processing: {split_folder}...")
    for img_id, info in images_info.items():
        width, height = info['width'], info['height']
        file_name = info['file_name']

        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations_by_image[img_id]:
            for seg in ann['segmentation']:
                if len(seg) >= 6: # Musi być min 3 punkty (x,y), żeby był trójkąt
                    poly = np.array(seg).reshape((int(len(seg)/2), 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], 255)

        safe_name = file_name.replace(".jpg", ".png").replace(".JPG", ".png")
        mask_filename = "mask_" + safe_name
        
        mask_path = os.path.join(split_folder, mask_filename)
        cv2.imwrite(mask_path, mask)
        
    print(f"Finished mask processing for {split_folder}!")

print("1. Downloading data from Roboflow...")
load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not API_KEY:
    raise ValueError("NO ROBOFLOW_API_KEY in .env")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("university-bwrcv").project("glenda")
version = project.version(1)

dataset = version.download("coco-segmentation")
print(f"Data downloaded to: {dataset.location}")

print("\n2. Converting COCO adnotations to PNG binary masks ...")

dataset_path = dataset.location 

for split in ["train", "valid", "test"]:
    split_dir = os.path.join(dataset_path, split)
    if os.path.exists(split_dir):
        convert_coco_to_masks(split_dir)
        
print("\nSucces!")