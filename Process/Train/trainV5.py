import torch
import random
import numpy as np
from ultralytics import YOLO
import os

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 调用种子设置函数
set_seed(42)

# 根目录列表，依次包含每个数据集根目录
root_dirs = [
    r"D:\YOLO_Benchmark\medical\brain-tumor",
    # 可以继续添加其他根目录
]

# 循环遍历每个根目录，逐个进行训练
for root_dir in root_dirs:
    # 数据集 YAML 文件路径
    data_yaml_path = os.path.join(root_dir, "data.yaml")
    
    # 提取当前根目录最后一个文件夹名称
    last_folder = os.path.basename(root_dir)
    
    # 设置保存路径和项目名称
    project_root = f"D:/YOLO_Benchmark/saved_models/{last_folder}"
    project_name = f"{last_folder}_yolo5m"

    # 加载模型
    model = YOLO(r"D:\YOLO_Benchmark\ultralytics\ultralytics\cfg\models\v5\yolov5.yaml")
    model = YOLO("yolov5mu.pt")  # 载入预训练模型

    # 训练模型
    train_results = model.train(
        data=data_yaml_path,
        epochs=300,               # 训练轮数
        imgsz=640,                # 训练图像大小
        batch=32,                 # 批次大小
        device="0",              # 使用GPU (0 表示第一个GPU)
        lr0=0.01,                 # 初始学习率
        augment=True,             # 启用数据增强
        translate=0.1,            # 设置平移增强
        scale=0.5,                # 设置缩放增强
        fliplr=0.5,               # 水平翻转概率
        hsv_h=0.015,              # 色相抖动
        hsv_s=0.7,                # 饱和度抖动
        hsv_v=0.4,                # 亮度抖动
        mosaic=True,              # 启用马赛克数据增强
        project=project_root,     # 动态设置保存项目的根目录
        name=project_name,        # 动态设置项目名称
        workers=0,                # 单过程数据加载
        save=True,               
        save_period=0             
    )

    print(f"训练完成: {project_name}")

# 各版本训练脚本调用
# python ultralytics/trainV5.py
# python ultralytics/trainV8.py
# python ultralytics/trainV9.py
# python ultralytics/trainV10.py
# python ultralytics/trainV11.py