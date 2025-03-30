import os
import torch
import pandas as pd
from ultralytics import YOLO
from thop import profile  # For calculating FLOPs and parameter count

# Set root directory and dataset YAML path
root_dir = r"D:\YOLO_Benchmark\saved_models\BCCD"
data_yaml = r"D:\YOLO_Benchmark\medical\BCCD\BCCD\data.yaml"  # Modify as needed

# List to store all evaluation results
all_results = []

# Walk through all subdirectories under the root directory
for subdir, _, files in os.walk(root_dir):
    # Check if 'weights/best.pt' exists in the directory
    if 'weights' in subdir and 'best.pt' in files:
        model_path = os.path.join(subdir, 'best.pt')
        model_name = os.path.basename(os.path.dirname(subdir))

        print(f"Evaluating model: {model_name} - Path: {model_path}")

        # Load the YOLO model
        model = YOLO(model_path)

        # Calculate parameter count
        params = sum(p.numel() for p in model.model.parameters())
        params_in_million = params / 1e6

        # Calculate FLOPs
        input = torch.randn(1, 3, 640, 640)
        flops, _ = profile(model.model, inputs=(input,), verbose=False)
        flops_in_gflops = flops / 1e9

        # Run evaluation on the test set
        test_results = model.val(
            data=data_yaml,
            split='test',
            imgsz=640,
            device="cuda",
            workers=0
        )

        # Get overall metrics
        mp, mr, map50, map50_95 = test_results.box.mean_results()
        class_maps = test_results.box.maps  # Per-class mAP array

        # Define class groups for small, medium, and large objects
        small_classes = [0]   # e.g., Platelets
        medium_classes = [1]  # e.g., RBC
        large_classes = [2]   # e.g., WBC

        # Compute mAP for each group
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

        # Collect results
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

        all_results.append(detailed_results)

        # Print the current model's results
        print(f"{model_name} Evaluation Results:")
        print(pd.DataFrame([detailed_results]))
        print("\n" + "-" * 50 + "\n")

# Save results if any valid models were found
if all_results:
    final_results_df = pd.DataFrame(all_results)
    save_dir = r"D:\YOLO_Benchmark\saved_evaluation"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"yolo_models_evaluation_{os.path.basename(root_dir.rstrip(os.sep))}.xlsx")
    final_results_df.to_excel(save_path, index=False)
    print(f"All model evaluation results have been saved to: {save_path}")
else:
    print("No valid model files found for evaluation.")
