import os
import argparse
from ultralytics import YOLO
import json
import cv2  # For checking image dimensions

def parse_args():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description='YOLO prediction with configurable paths')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Root directory containing model weights (e.g., D:/YOLO_Benchmark/saved_models/UWD)')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to data.yaml file (e.g., D:/YOLO_Benchmark/datasets2/UWD/data.yaml)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save prediction results (e.g., D:/YOLO_Benchmark/model_result/BDD100k)')
    parser.add_argument('--max_size', type=int, default=10,
                        help='Maximum image size in MB to process (default: 10MB)')  # Set the threshold size

    return parser.parse_args()

def is_image_too_large(image_path, max_size_mb):
    """
    Check if the image is too large (based on file size).
    """
    image_size = os.path.getsize(image_path) / (2000 * 2000)  # Size in MB
    return image_size > max_size_mb

def main():
    # Clear CUDA cache
    import torch
    torch.cuda.empty_cache()

    args = parse_args()

    # Walk through the directory to find models
    for subdir, _, files in os.walk(args.root_dir):
        if 'weights' in subdir and 'best.pt' in files:
            model_path = os.path.join(subdir, 'best.pt')
            model_name = os.path.basename(os.path.dirname(subdir))

            # Load model
            model = YOLO(model_path)

            # Prediction
            image_dir = os.path.join(os.path.dirname(args.data_yaml), "test/images")
            results = []

            for image_name in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_name)

                # Skip large images
                if is_image_too_large(image_path, args.max_size):
                    print(f"Skipping {image_name}, image is too large.")
                    continue

                # Perform prediction for the valid image
                result = model.predict(
                    source=image_path,
                    imgsz=640,
                    device="cuda",
                    batch=1,
                    save=False
                )

                # Process prediction results
                for r in result:
                    image_id = os.path.splitext(image_name)[0]
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
                            results.append(pred_dict)

            # Save prediction results
            os.makedirs(args.output_dir, exist_ok=True)
            pred_save_path = os.path.join(args.output_dir, f"{model_name}.json")
            with open(pred_save_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Predictions saved for model {model_name} at {pred_save_path}")

if __name__ == "__main__":
    main()
