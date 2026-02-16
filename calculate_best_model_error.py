#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate scale bar length error of the best model
"""

import os
import pandas as pd
import torch
from ultralytics import YOLO

def calculate_scale_length_error(model_path, gt_csv='scale_bar_labels_fixed.csv', val_images_dir='yolo11_optimized_dataset/images/val'):
    """Calculate scale bar length error"""
    try:
        # Read ground truth labels
        gt_df = pd.read_csv(gt_csv)
        gt_df['true_length'] = abs(gt_df['x2'] - gt_df['x1'])  # Use horizontal coordinate difference
        
        # Check validation set path
        if not os.path.exists(val_images_dir):
            print(f" Validation set path does not exist: {val_images_dir}")
            return float('inf')
        
        # Load model
        print(f" Loading model: {model_path}")
        model = YOLO(model_path)
        
        total_error = 0
        valid_predictions = 0
        
        # Predict on validation set
        print(" Starting to calculate scale bar length error...")
        for img_file in os.listdir(val_images_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(val_images_dir, img_file)
            results = model.predict(img_path, conf=0.5, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get predicted bounding box
                pred_box = results[0].boxes.xyxy[0].cpu().numpy()
                pred_length = abs(pred_box[2] - pred_box[0])  # Use horizontal coordinate difference
                
                # Find corresponding ground truth label
                img_name = img_file
                gt_match = gt_df[gt_df['filename'] == img_name]
                
                if len(gt_match) > 0:
                    true_length = gt_match.iloc[0]['true_length']
                    if true_length > 0:
                        relative_error = abs(pred_length - true_length) / true_length * 100
                        total_error += relative_error
                        valid_predictions += 1
                        print(f"   {img_file}: Prediction={pred_length:.1f}px, Ground Truth={true_length:.1f}px, Error={relative_error:.2f}%")
        
        # Return average relative error
        if valid_predictions > 0:
            avg_error = total_error / valid_predictions
            print(f"\n Statistical Results:")
            print(f"   Number of valid predictions: {valid_predictions}")
            print(f"   Average relative error: {avg_error:.2f}%")
            return avg_error
        else:
            print(" No valid prediction results")
            return float('inf')
            
    except Exception as e:
        print(f" Error calculating length error: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

def main():
    """Main function"""
    print(" Calculate scale bar length error of the best model")
    print("="*50)
    
    # Best model path
    best_model_path = "yolo11_optimized_runs/optimized_scalebar_detection3/weights/best.pt"
    
    if not os.path.exists(best_model_path):
        print(f" Best model file does not exist: {best_model_path}")
        return
    
    # Calculate error
    error = calculate_scale_length_error(best_model_path)
    
    print(f"\n Final Results:")
    print(f"   Model: {best_model_path}")
    print(f"   Scale bar length error: {error:.2f}%")
    
    if error <= 5.0:
        print(f" Excellent! Error: {error:.2f}% <= 5%")
    elif error <= 10.0:
        print(f" Good! Error: {error:.2f}% <= 10%")
    else:
        print(f" Further optimization is needed, current error: {error:.2f}%")

if __name__ == '__main__':
    main()
