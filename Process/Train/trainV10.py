import torch
import random
import numpy as np
from ultralytics import YOLO
import os

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# List of dataset root directories
root_dirs = [
    r"D:\YOLO_Benchmark\medical\brain-tumor",
    # Add more dataset directories if needed
]

# Loop through each dataset root
for root_dir in root_dirs:
    # Path to dataset YAML
    data_yaml_path = os.path.join(root_dir, "data.yaml")
    
    # Extract dataset name
    last_folder = os.path.basename(root_dir)
    
    # Set project directory and experiment name
    project_root = f"D:/YOLO_Benchmark/saved_models/{last_folder}"
    project_name = f"{last_folder}_yolo10m"

    # Load model architecture and pretrained weights
    model = YOLO(r"D:\YOLO_Benchmark\ultralytics\ultralytics\cfg\models\v10\yolov10m.yaml")
    model = YOLO("yolov10m.pt")  # Load pretrained model

    # Train the model
    train_results = model.train(
        data=data_yaml_path,
        epochs=300,
        imgsz=640,
        batch=32,
        device="0",               # GPU index
        lr0=0.01,
        augment=True,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=True,
        project=project_root,
        name=project_name,
        workers=0,
        save=True,
        save_period=0
    )

    print(f"Training completed: {project_name}")

# Other version training scripts:
# python ultralytics/trainV5.py
# python ultralytics/trainV8.py
# python ultralytics/trainV9.py
# python ultralytics/trainV10.py
# python ultralytics/trainV11.py
