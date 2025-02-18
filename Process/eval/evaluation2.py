import os
import torch
import pandas as pd
from ultralytics import YOLO
from thop import profile  # 用于计算 FLOPs 和参数数量

# 设置根目录和数据集的 YAML 文件路径
root_dir = r"D:\YOLO_Benchmark\saved_models\BCCD"
data_yaml = r"D:\YOLO_Benchmark\medical\BCCD\BCCD\data.yaml"  # 请根据实际情况修改

# 存储所有评估结果的列表
all_results = []

# 遍历根目录下的所有子目录
for subdir, _, files in os.walk(root_dir):
    # 检查子目录是否包含 weights/best.pt 文件
    if 'weights' in subdir and 'best.pt' in files:
        model_path = os.path.join(subdir, 'best.pt')
        
        # 获取模型名称（例如 "BCCD_yolo8n"）
        model_name = os.path.basename(os.path.dirname(subdir))

        # 打印模型名称和路径
        print(f"正在评估模型: {model_name} - 路径: {model_path}")

        # 加载模型
        model = YOLO(model_path)

        # 计算参数数量
        params = sum(p.numel() for p in model.model.parameters())
        params_in_million = params / 1e6  # 参数数量（百万）

        # 计算 FLOPs
        input = torch.randn(1, 3, 640, 640)  # 根据模型输入尺寸调整
        flops, _ = profile(model.model, inputs=(input, ), verbose=False)
        flops_in_gflops = flops / 1e9  # FLOPs（GFLOPs）

        # 在测试集上进行评估
        test_results = model.val(
            data=data_yaml,
            split='test',
            imgsz=640,
            device="cuda",  # 使用 GPU
            workers=0
        )

        # 从 box 对象中获取评估结果，包括 mAP@50 和 mAP@50-95
        mp, mr, map50, map50_95 = test_results.box.mean_results()

        # 获取每个类的 mAP
        class_maps = test_results.box.maps  # maps 是一个 NumPy 数组，包含每个类别的 mAP 值

        # 假设类别已经按照 Small, Medium, Large 分组
        small_classes = [0]  # Small 类别的索引，例如 "Platelets"
        medium_classes = [1]  # Medium 类别的索引，例如 "RBC"
        large_classes = [2]  # Large 类别的索引，例如 "WBC"

        # 计算 Small, Medium, Large 的 mAP
        map_small = (
            sum(class_maps[c] for c in small_classes if c < len(class_maps)) / len(small_classes)
            if small_classes else None
        )
        map_medium = (
            sum(class_maps[c] for c in medium_classes if c < len(class_maps)) / len(medium_classes)
            if medium_classes else None
        )
        map_large = (
            sum(class_maps[c] for c in large_classes if c < len(class_maps)) / len(large_classes)
            if large_classes else None
        )

        # 保存当前模型的评估结果
        detailed_results = {
            "Model": model_name,
            "Mean Precision": mp,
            "Mean Recall": mr,
            "mAP@50": map50,
            "mAP@75": test_results.box.maps[2] if len(test_results.box.maps) > 2 else None,
            "mAP@50-95": map50_95,
            "mAP Small": map_small,
            "mAP Medium": map_medium,
            "mAP Large": map_large,
            "Parameters (M)": params_in_million,
            "FLOPs (GFLOPs)": flops_in_gflops
        }
        
        # 将当前模型的评估结果添加到总结果列表
        all_results.append(detailed_results)

        # 输出评估结果
        print(f"{model_name} 测试集评估结果:")
        print(pd.DataFrame([detailed_results]))
        print("\n" + "-" * 50 + "\n")

# 如果找到有效的模型评估结果，则合并并保存
if all_results:
    final_results_df = pd.DataFrame(all_results)
    save_dir = r"D:\YOLO_Benchmark\saved_evaluation"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"yolo_models_evaluation_{os.path.basename(root_dir.rstrip(os.sep))}.xlsx")
    final_results_df.to_excel(save_path, index=False)
    print(f"所有模型的评估结果已保存至: {save_path}")
else:
    print("未找到任何符合条件的模型文件进行评估。")
### small and v6 adjustment complete happy I have a good day