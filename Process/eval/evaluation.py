import os
import pandas as pd
from ultralytics import YOLO

# 设置根目录和数据集的 YAML 文件路径
root_dir = r"D:\YOLO_Benchmark\saved_models\brain-tumor"
data_yaml = r"D:\YOLO_Benchmark\medical\brain-tumor\data.yaml"  # 请根据实际情况修改

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
        
        # 直接访问 maps 属性，该属性为 numpy 数组
        maps = test_results.box.maps
        
        # 单独记录 mAP@75
        map75 = maps[2] if len(maps) > 2 else None  # 索引 2 可能代表 mAP@75

        # 保存当前模型的评估结果
        detailed_results = {
            "Model": model_name,
            "Mean Precision": mp,
            "Mean Recall": mr,
            "mAP@50": map50,
            "mAP@75": map75,
            "mAP@50-95": map50_95
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
