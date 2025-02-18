from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np
import glob
import os
import csv

def evaluate_single_model(gt_file, pred_file):
    try:
        # Load ground truth
        cocoGt = COCO(gt_file)
        
        # Load predictions
        with open(pred_file, 'r') as f:
            predictions = json.load(f)
            
        # Process predictions
        processed_predictions = []
        for pred in predictions:
            pred_copy = {
                'image_id': pred['image_id'],
                'category_id': int(pred['category_id']),
                'bbox': [float(x) for x in pred['bbox']],
                'score': float(pred['score'])
            }
            if pred_copy['score'] > 1:
                pred_copy['score'] = pred_copy['score'] / 100
            processed_predictions.append(pred_copy)
        
        # Load predictions into COCO format
        cocoDt = cocoGt.loadRes(processed_predictions)
        
        # Create evaluator
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        
        # Calculate mAP@0.5
        cocoEval.params.iouThrs = [0.5]  # Only IoU=0.5
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map_50 = cocoEval.stats[0]  # mAP@0.5
        
        # Reset for default IoU range (0.5:0.95)
        cocoEval.params.iouThrs = np.linspace(.5, .95, int(np.round((0.95 - 0.5) / .05)) + 1, endpoint=True)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # Get mAP@[.50:.95] and MAPs for different sizes
        map_50_95 = cocoEval.stats[0]  # mAP@[.50:.95]
        map_small = cocoEval.stats[3]   # MAP for small objects
        map_medium = cocoEval.stats[4]  # MAP for medium objects
        map_large = cocoEval.stats[5]   # MAP for large objects
        
        return [map_50, map_50_95, map_small, map_medium, map_large]
    
    except Exception as e:
        print(f"Error evaluating {pred_file}: {str(e)}")
        return [0, 0, 0, 0, 0]

def main():
    # Get all prediction JSON files
    pred_files = glob.glob("*.json")
    gt_file = "instances_test.json"
    
    # Remove ground truth file from prediction files
    if gt_file in pred_files:
        pred_files.remove(gt_file)
    
    # Prepare results
    results = []
    headers = ['Model', 'mAP@0.5', 'mAP@[.50:.95]', 'MAP (small)', 'MAP (medium)', 'MAP (large)']
    
    # Evaluate each model
    for pred_file in pred_files:
        model_name = os.path.splitext(pred_file)[0]  # Get filename without extension
        print(f"\nEvaluating {model_name}...")
        
        aps = evaluate_single_model(gt_file, pred_file)
        
        # Print results
        print(f"Results for {model_name}:")
        for ap_type, ap_value in zip(headers[1:], aps):
            print(f"{ap_type}: {ap_value:.4f}")
        
        # Store results
        results.append([model_name] + [f"{ap:.4f}" for ap in aps])
    
    # Save to CSV
    output_file = 'evaluation_results.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()