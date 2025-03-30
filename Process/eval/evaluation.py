import os
import pandas as pd
from ultralytics import YOLO

# Set the root directory and the path to the dataset's YAML file
root_dir = r"D:\YOLO_Benchmark\saved_models\brain-tumor"
data_yaml = r"D:\YOLO_Benchmark\medical\brain-tumor\data.yaml"  # Modify this path as needed

# A list to store all evaluation results
all_results = []

# Traverse all subdirectories under the root directory
for subdir, _, files in os.walk(root_dir):
    # Check if the subdirectory contains weights/best.pt
    if 'weights' in subdir and 'best.pt' in files:
        model_path = os.path.join(subdir, 'best.pt')
        
        # Extract the model name (e.g., "BCCD_yolo8n")
        model_name = os.path.basename(os.path.dirname(subdir))

        # Print model name and path
        print(f"Evaluating model: {model_name} - Path: {model_path}")

        # Load the model
        model = YOLO(model_path)

        # Evaluate on the test set
        test_results = model.val(
            data=data_yaml,
            split='test',
            imgsz=640,
            device="cuda",  # Use GPU
            workers=0
        )

        # Retrieve evaluation results from the box object, including mAP@50 and mAP@50-95
        mp, mr, map50, map50_95 = test_results.box.mean_results()
        
        # Directly access the maps attribute, which is a NumPy array
        maps = test_results.box.maps
        
        # Record mAP@75 separately
        map75 = maps[2] if len(maps) > 2 else None  # Index 2 might represent mAP@75

        # Save the evaluation results for the current model
        detailed_results = {
            "Model": model_name,
            "Mean Precision": mp,
            "Mean Recall": mr,
            "mAP@50": map50,
            "mAP@75": map75,
            "mAP@50-95": map50_95
        }
        
        # Add the current model's results to the overall list
        all_results.append(detailed_results)

        # Print the evaluation results
        print(f"{model_name} test set evaluation results:")
        print(pd.DataFrame([detailed_results]))
        print("\n" + "-" * 50 + "\n")

# If there are valid model evaluation results, combine and save them
if all_results:
    final_results_df = pd.DataFrame(all_results)
    save_dir = r"D:\YOLO_Benchmark\saved_evaluation"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"yolo_models_evaluation_{os.path.basename(root_dir.rstrip(os.sep))}.xlsx")
    final_results_df.to_excel(save_path, index=False)
    print(f"All model evaluation results have been saved to: {save_path}")
else:
    print("No valid model files found for evaluation.")
