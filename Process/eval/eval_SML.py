# import os
# from ultralytics import YOLO
# import json

# root_dir = r"D:\YOLO_Benchmark\saved_models\UWD"
# data_yaml = r"D:\YOLO_Benchmark\datasets2\UWD\data.yaml"

# for subdir, _, files in os.walk(root_dir):
#    if 'weights' in subdir and 'best.pt' in files:
#        model_path = os.path.join(subdir, 'best.pt')
#        model_name = os.path.basename(os.path.dirname(subdir))
       
#        # Add model loading here
#        model = YOLO(model_path)
       
#        results = model.predict(
#            source=os.path.join(os.path.dirname(data_yaml), "test/images"),
#            imgsz=640,
#            device="cuda",
#            save=False
#        )
       
#        predictions = []
#        for r in results:
#            image_id = os.path.splitext(os.path.basename(r.path))[0]
#            if len(r.boxes):
#                for box in r.boxes:
#                    x1, y1, x2, y2 = box.xyxy[0].tolist()
#                    w = x2 - x1
#                    h = y2 - y1
#                    pred_dict = {
#                        "image_id": image_id,
#                        "category_id": int(box.cls[0]),
#                        "bbox": [float(x1), float(y1), float(w), float(h)],
#                        "score": float(box.conf[0])
#                    }
#                    predictions.append(pred_dict)

#        model_results_dir = r"D:\YOLO_Benchmark\model_result\UWD"
#        os.makedirs(model_results_dir, exist_ok=True)
#        pred_save_path = os.path.join(model_results_dir, f"{model_name}.json")
#        with open(pred_save_path, 'w') as f:
#            json.dump(predictions, f, indent=2)


import os
import argparse
from ultralytics import YOLO
import json

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='YOLO prediction with configurable paths')
    parser.add_argument('--root_dir', type=str, required=True,
                      help='Root directory containing model weights (e.g., D:/YOLO_Benchmark/saved_models/UWD)')
    parser.add_argument('--data_yaml', type=str, required=True,
                      help='Path to data.yaml file (e.g., D:/YOLO_Benchmark/datasets2/UWD/data.yaml)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save prediction results (e.g., D:/YOLO_Benchmark/model_result/UWD)')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    import torch
    torch.cuda.empty_cache()

    args = parse_args()
    
    # 遍历目录查找模型
    for subdir, _, files in os.walk(args.root_dir):
        if 'weights' in subdir and 'best.pt' in files:
            model_path = os.path.join(subdir, 'best.pt')
            model_name = os.path.basename(os.path.dirname(subdir))
            
            # 加载模型
            model = YOLO(model_path)
            
            # 预测
            results = model.predict(
                source=os.path.join(os.path.dirname(args.data_yaml), "test/images"),
                imgsz=640,
                device="cuda",
                batch=1,
                save=False
            )
            
            # 处理预测结果
            predictions = []
            for r in results:
                image_id = os.path.splitext(os.path.basename(r.path))[0]
                if len(r.boxes):
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w = x2 - x1
                        h = y2 - y1
                        pred_dict = {
                            "image_id": image_id,
                            "category_id": int(box.cls[0]),
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "score": float(box.conf[0])
                        }
                        predictions.append(pred_dict)
            
            # 保存结果
            os.makedirs(args.output_dir, exist_ok=True)
            pred_save_path = os.path.join(args.output_dir, f"{model_name}.json")
            with open(pred_save_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            print(f"Predictions saved for model {model_name} at {pred_save_path}")

if __name__ == "__main__":
    main()