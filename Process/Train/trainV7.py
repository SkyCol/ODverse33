import os
import subprocess
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Root dataset directories
root_dirs = [
    r"D:\YOLO_Benchmark\medical\brain-tumor",
    # Add more if needed
]

# YOLOv7 training script path
yolov7_dir = r"D:\YOLO_Benchmark\yolov7"  # path to cloned yolov7 repo
train_script = os.path.join(yolov7_dir, "train.py")

# Loop over each dataset
for root_dir in root_dirs:
    data_yaml = os.path.join(root_dir, "data.yaml")
    last_folder = os.path.basename(root_dir)
    project_name = f"{last_folder}_yolo7"
    save_dir = f"D:/YOLO_Benchmark/saved_models/{last_folder}"

    command = [
        "python", train_script,
        "--weights", "yolov7.pt",
        "--cfg", "cfg/training/yolov7.yaml",
        "--data", data_yaml,
        "--device", "0",
        "--batch-size", "32",
        "--epochs", "300",
        "--img", "640",
        "--project", save_dir,
        "--name", project_name,
        "--exist-ok"
    ]

    print(f"Training: {project_name}")
    subprocess.run(command, cwd=yolov7_dir)
    print(f"Completed: {project_name}")
