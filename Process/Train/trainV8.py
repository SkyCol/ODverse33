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

# Apply the seed
set_seed(42)

# List of dataset root directories
root_dirs = [
    r"D:\YOLO_Benchmark\medical\bone-fracture\bone-fracture",
    # Add more dataset root directories if needed
]

# Loop through each dataset directory and start training
for root_dir in root_dirs:
    # Path to the dataset configuration (data.yaml)
    data_yaml_path = os.path.join(root_dir, "data.yaml")

    # Get the last folder name to use as an identifier
    last_folder = os.path.basename(root_dir)

    # Define where to save the model and logs
    project_root = f"D:/YOLO_Benchmark/saved_models/{last_folder}"
    project_name = f"{last_folder}_yolo8m"

    # Load the pretrained YOLOv8 model (medium version)
    model = YOLO("yolov8m.pt")

    # Start training the model
    train_results = model.train(
        data=data_yaml_path,
        epochs=300,               # Number of training epochs
        imgsz=640,                # Input image size
        batch=32,                 # Batch size
        device="0",               # GPU index (0 for the first GPU)
        lr0=0.01,                 # Initial learning rate
        augment=True,             # Enable data augmentation
        translate=0.1,            # Random translation
        scale=0.5,                # Random scaling
        fliplr=0.5,               # Horizontal flip probability
        hsv_h=0.015,              # HSV hue augmentation
        hsv_s=0.7,                # HSV saturation augmentation
        hsv_v=0.4,                # HSV value augmentation
        mosaic=True,              # Enable mosaic augmentation
        project=project_root,     # Directory to save results
        name=project_name,        # Project name
        workers=0,                # Number of dataloader workers
        save=True,                # Save checkpoints
        save_period=0             # Save every epoch (0 means only save best and last)
    )

    print(f"Training completed: {project_name}")
