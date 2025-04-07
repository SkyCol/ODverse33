import os
import json
from ultralytics import YOLO

def save_predictions_as_txt(model, data_yaml, output_dir, model_name):
    """
    使用 YOLO 模型对数据集进行预测，并将结果保存为单独的 txt 文件。
    参数：
        model: YOLO 模型对象。
        data_yaml: 数据集 YAML 文件的路径。
        output_dir: 保存预测结果的目录。
        model_name: 当前模型的名称，用于区分不同模型的预测结果。
    """
    # 使用 YOLO 内置验证进行评估
    test_results = model.val(
        data=data_yaml,
        split='test',
        imgsz=640,
        device="cuda",  # 使用 GPU
        workers=0,
        save_json=True  # 保存详细结果为 COCO 格式的 JSON 文件
    )

    # 确定 predictions.json 文件的路径
    predictions_path = os.path.join(test_results.save_dir, "predictions.json")
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"预测结果文件未找到：{predictions_path}")

    # 加载预测结果
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    # 创建模型特定的保存目录
    model_output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    # 将预测结果保存为单独的 txt 文件
    for pred in predictions:
        image_id = pred['image_id']
        category_id = pred['category_id']
        score = pred['score']
        bbox = pred['bbox']  # [x_min, y_min, width, height]

        # 转换为 YOLO 格式: [center_x, center_y, width, height]
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        width = bbox[2]
        height = bbox[3]

        # 构造保存路径
        file_path = os.path.join(model_output_dir, f"{image_id}.txt")

        # 写入预测结果
        with open(file_path, 'a') as f:
            f.write(f"{category_id} {score:.6f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # 打印 mAP50
    try:
        map_50 = test_results.box.map  # 获取 mAP50 值
        print(f"mAP@50: {map_50:.3f}")
    except AttributeError:
        print("无法获取 mAP@50 值，可能是 YOLO 版本更新导致。")

# 主程序
if __name__ == "__main__":
    # 设置路径
    root_dir = r"D:\\YOLO_Benchmark\\saved_models\\UWD"  # 模型保存目录
    data_yaml = r"D:\YOLO_Benchmark\datasets2\UWD\data.yaml"  # 数据集 YAML 文件路径
    output_dir = r"D:\\YOLO_Benchmark\\test_predictions"  # 保存预测标签的目录

    # 遍历目录寻找 YOLO 模型
    for subdir, _, files in os.walk(root_dir):
        if 'weights' in subdir and 'best.pt' in files:
            model_path = os.path.join(subdir, 'best.pt')
            model_name = os.path.basename(os.path.dirname(subdir))

            print(f"正在评估模型：{model_name} - 路径：{model_path}")

            # 加载模型
            model = YOLO(model_path)

            # 保存预测结果
            save_predictions_as_txt(model, data_yaml, output_dir, model_name)

            print(f"预测结果已保存至目录：{os.path.join(output_dir, model_name)}")
            print("\n" + "-" * 50 + "\n")
### first i need to complete 2 task
### compared dataset
### predict SML   