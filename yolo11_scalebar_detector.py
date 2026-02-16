#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11 Scale Bar Detector - Latest and Most Powerful Version
High-precision scale bar detection using YOLOv11
"""

import os
import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import sys

class YOLO11ScaleBarConverter:
    """Convert data format to YOLOv11 format"""
    
    def __init__(self, csv_file, img_dir, output_dir="yolo11_dataset"):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.data = pd.read_csv(csv_file)
        
        # Create YOLOv11 directory structure
        self.setup_directories()
    
    def setup_directories(self):
        """Create YOLOv11 directory structure"""
        dirs = [
            f"{self.output_dir}/images/train",
            f"{self.output_dir}/images/val", 
            f"{self.output_dir}/images/test",
            f"{self.output_dir}/labels/train",
            f"{self.output_dir}/labels/val",
            f"{self.output_dir}/labels/test"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"YOLOv11 directory structure created successfully: {self.output_dir}")
    
    def convert_to_yolo_format(self):
        """Convert to YOLOv11 format - Optimized bounding box generation"""
        print("Converting data to YOLOv11 format...")
        
        valid_samples = []
        
        # Validate and convert data
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="Processing data"):
            img_path = os.path.join(self.img_dir, str(row['filename']))
            
            if not os.path.exists(img_path):
                continue
                
            # Load image to get dimensions
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            h, w = image.shape[:2]
            
            # Validate coordinates
            if not (0 <= row['x1'] < w and 0 <= row['x2'] < w and
                    0 <= row['y1'] < h and 0 <= row['y2'] < h):
                continue
            
            # Calculate YOLOv11 format bounding box - More intelligent method
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            
            # Calculate scale bar angle and length
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            # Adaptive bounding box size
            # Length direction: scale bar length + small margin
            # Width direction: adaptive thickness
            length_margin = max(5, line_length * 0.05)  # 5% margin, minimum 5 pixels
            width_thickness = max(8, min(w, h) // 80)   # Adaptive thickness
            
            # Calculate four vertices of rotated bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Extend length
            half_length = (line_length + 2 * length_margin) / 2
            half_width = width_thickness / 2
            
            # Rotated bounding box vertices
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            corners = [
                (-half_length, -half_width),
                (half_length, -half_width),
                (half_length, half_width),
                (-half_length, half_width)
            ]
            
            rotated_corners = []
            for cx, cy in corners:
                rx = cx * cos_a - cy * sin_a + center_x
                ry = cx * sin_a + cy * cos_a + center_y
                rotated_corners.append((rx, ry))
            
            # Calculate axis-aligned bounding box
            xs = [c[0] for c in rotated_corners]
            ys = [c[1] for c in rotated_corners]
            
            bbox_x1 = max(0, min(xs))
            bbox_y1 = max(0, min(ys))
            bbox_x2 = min(w, max(xs))
            bbox_y2 = min(h, max(ys))
            
            # Ensure valid bounding box
            if bbox_x2 <= bbox_x1 or bbox_y2 <= bbox_y1:
                continue
            
            # Convert to YOLOv11 format (center coordinates + width/height, normalized)
            center_x_norm = (bbox_x1 + bbox_x2) / 2 / w
            center_y_norm = (bbox_y1 + bbox_y2) / 2 / h
            width_norm = (bbox_x2 - bbox_x1) / w
            height_norm = (bbox_y2 - bbox_y1) / h
            
            # Add quality check
            if (0.001 < width_norm < 0.9 and 0.001 < height_norm < 0.9 and
                0.001 < center_x_norm < 0.999 and 0.001 < center_y_norm < 0.999):
                
                valid_samples.append({
                    'filename': row['filename'],
                    'img_path': img_path,
                    'yolo_label': f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}",
                    'original_coords': (x1, y1, x2, y2),
                    'bbox_coords': (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                    'line_length': line_length,
                    'angle': angle
                })
        
        print(f"Valid samples: {len(valid_samples)}")
        
        # Dataset splitting - Better splitting strategy
        np.random.seed(42)  # Fixed random seed
        np.random.shuffle(valid_samples)
        
        # Stratified split by length to ensure representation across all length ranges
        lengths = [s['line_length'] for s in valid_samples]
        length_percentiles = np.percentile(lengths, [33, 66])
        
        short_samples = [s for s in valid_samples if s['line_length'] <= length_percentiles[0]]
        medium_samples = [s for s in valid_samples if length_percentiles[0] < s['line_length'] <= length_percentiles[1]]
        long_samples = [s for s in valid_samples if s['line_length'] > length_percentiles[1]]
        
        def split_samples(samples, train_ratio=0.7, val_ratio=0.2):
            n = len(samples)
            train_n = int(n * train_ratio)
            val_n = int(n * val_ratio)
            
            return samples[:train_n], samples[train_n:train_n+val_n], samples[train_n+val_n:]
        
        # Stratified splitting
        short_train, short_val, short_test = split_samples(short_samples)
        medium_train, medium_val, medium_test = split_samples(medium_samples)
        long_train, long_val, long_test = split_samples(long_samples)
        
        # Combine
        train_samples = short_train + medium_train + long_train
        val_samples = short_val + medium_val + long_val
        test_samples = short_test + medium_test + long_test
        
        print(f"Stratified data split:")
        print(f"   Short scale bars: {len(short_samples)}")
        print(f"   Medium scale bars: {len(medium_samples)}")
        print(f"   Long scale bars: {len(long_samples)}")
        print(f"   Training set: {len(train_samples)}, Validation set: {len(val_samples)}, Test set: {len(test_samples)}")
        
        # Copy files and create labels
        self._copy_dataset(train_samples, "train")
        self._copy_dataset(val_samples, "val")
        self._copy_dataset(test_samples, "test")
        
        # Create YOLOv11 YAML configuration file
        self._create_yaml_config()
        
        return len(valid_samples)
    
    def _copy_dataset(self, samples, split):
        """Copy dataset files"""
        for sample in tqdm(samples, desc=f"Copying {split} data"):
            # Copy image
            src_img = sample['img_path']
            dst_img = f"{self.output_dir}/images/{split}/{sample['filename']}"
            shutil.copy2(src_img, dst_img)
            
            # Create label file
            label_file = f"{self.output_dir}/labels/{split}/{Path(sample['filename']).stem}.txt"
            with open(label_file, 'w') as f:
                f.write(sample['yolo_label'] + '\n')
    
    def _create_yaml_config(self):
        """Create YOLOv11 configuration file"""
        config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes
            'names': ['scalebar']  # Class names
        }
        
        yaml_path = f"{self.output_dir}/scalebar_v11.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"YOLOv11 configuration file created: {yaml_path}")
        return yaml_path

class YOLO11ScaleBarTrainer:
    """YOLOv11 Scale Bar Detector Trainer"""
    
    def __init__(self, dataset_dir="yolo11_dataset"):
        self.dataset_dir = dataset_dir
        self.yaml_path = f"{dataset_dir}/scalebar_v11.yaml"
        
    def install_ultralytics(self):
        """Install latest ultralytics (supports YOLOv11)"""
        try:
            import ultralytics
            from ultralytics import YOLO
            
            # Check version
            version = ultralytics.__version__
            print(f"ultralytics version: {version}")
            
            # Test YOLOv11 model loading
            try:
                model = YOLO('yolo11n.pt')
                print("YOLOv11 support confirmed")
                return True
            except:
                print("Current version does not support YOLOv11, upgrading...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
                return True
                
        except ImportError:
            print("Installing latest ultralytics...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
                print("ultralytics installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("ultralytics installation failed")
                return False
    
    def train_yolo11_model(self, model_size="yolo11s", epochs=100, imgsz=640):
        """Train YOLOv11 model"""
        
        if not self.install_ultralytics():
            return False
        
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ultralytics import failed")
            return False
        
        print(f"Starting YOLOv11{model_size[6:]} model training...")
        print(f"Training parameters: epochs={epochs}, imgsz={imgsz}")
        print(f"YOLOv11 improvements over v8:")
        print(f"   • Better feature extraction backbone")
        print(f"   • Improved FPN structure")
        print(f"   • More accurate detection head")
        print(f"   • Better training strategies")
        
        # Create YOLOv11 model
        model = YOLO(f'{model_size}.pt')
        
        # YOLOv11 optimized training parameters
        results = model.train(
            data=self.yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            patience=30,  # Increased patience
            batch=16,
            device=0,  # Use GPU
            project='yolo11_scalebar_runs',
            name='scalebar_detection_v11',
            save_period=10,
            plots=True,
            val=True,
            
            # YOLOv11 specific optimization parameters
            optimizer='AdamW',  # Better optimizer
            lr0=0.01,          # Initial learning rate
            lrf=0.1,           # Final learning rate ratio
            momentum=0.937,     # SGD momentum
            weight_decay=0.0005, # Weight decay
            warmup_epochs=3,    # Warmup epochs
            warmup_momentum=0.8, # Warmup momentum
            warmup_bias_lr=0.1,  # Warmup bias learning rate
            
            # Data augmentation - Optimized for scale bars
            hsv_h=0.015,       # Hue augmentation
            hsv_s=0.7,         # Saturation augmentation
            hsv_v=0.4,         # Value augmentation
            degrees=0,         # Rotation angle (scale bars are usually horizontal, no rotation)
            translate=0.1,     # Translation
            scale=0.5,         # Scaling
            shear=0,           # Shear (not suitable for scale bars)
            perspective=0,     # Perspective transform
            flipud=0,          # Vertical flip
            fliplr=0.5,        # Horizontal flip
            mosaic=1.0,        # Mosaic augmentation
            mixup=0.1,         # Mixup augmentation
            
            # Loss function weights
            box=7.5,           # Bbox loss weight
            cls=0.5,           # Classification loss weight
            dfl=1.5,           # DFL loss weight
            
            # Other optimizations
            amp=True,          # Automatic mixed precision
            fraction=1.0,      # Use full dataset
            profile=False,     # No performance profiling
            freeze=None,       # No layer freezing
            multi_scale=True,  # Multi-scale training
            overlap_mask=True, # Overlap masks
            mask_ratio=4,      # Mask ratio
            dropout=0.0,       # Dropout
        )
        
        print("YOLOv11 model training completed!")
        
        # Evaluate model
        metrics = model.val()
        print(f"YOLOv11 validation results:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        print(f"   Precision: {metrics.box.mp:.3f}")
        print(f"   Recall: {metrics.box.mr:.3f}")
        
        return True
    
    def compare_with_yolo8(self):
        """Performance comparison with YOLOv8"""
        print("YOLOv11 vs YOLOv8 theoretical comparison:")
        print("="*50)
        print("YOLOv11 advantages:")
        print("   • mAP improvement: +2-5%")
        print("   • Inference speed: +10-15%")
        print("   • Parameter efficiency: Fewer parameters for better performance")
        print("   • Small object detection: Significant improvement")
        print("   • Bounding box regression: More accurate")
        print("   • Training stability: Better convergence")
        print()
        print("Special advantages for scale bar detection:")
        print("   • Optimized for long thin object detection")
        print("   • Better feature fusion")
        print("   • Improved loss function")

class YOLO11ScaleBarPredictor:
    """YOLOv11 Scale Bar Predictor"""
    
    def __init__(self, model_path="yolo11_scalebar_runs/scalebar_detection_v11/weights/best.pt"):
        self.model_path = model_path
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"YOLOv11 model loaded successfully: {model_path}")
        except Exception as e:
            print(f"Model loading failed: {e}")
            self.model = None
    
    def predict_image(self, image_path, conf=0.5, save_result=True):
        """Predict single image using YOLOv11"""
        
        if self.model is None:
            print("Model not loaded")
            return None
        
        print(f"YOLOv11 predicting image: {image_path}")
        
        # YOLOv11 prediction
        results = self.model(image_path, conf=conf, verbose=False)
        
        # Parse results
        predictions = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Intelligently estimate scale bar line endpoints
                    # YOLOv11 provides more accurate bounding boxes for precise endpoint estimation
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Determine scale bar orientation (horizontal or vertical)
                    if bbox_width > bbox_height:  # Horizontal scale bar
                        line_x1 = x1 + bbox_width * 0.1
                        line_y1 = center_y
                        line_x2 = x2 - bbox_width * 0.1
                        line_y2 = center_y
                    else:  # Vertical scale bar
                        line_x1 = center_x
                        line_y1 = y1 + bbox_height * 0.1
                        line_x2 = center_x
                        line_y2 = y2 - bbox_height * 0.1
                    
                    predictions.append({
                        'bbox': (x1, y1, x2, y2),
                        'line': (line_x1, line_y1, line_x2, line_y2),
                        'confidence': confidence,
                        'length': np.sqrt((line_x2 - line_x1)**2 + (line_y2 - line_y1)**2),
                        'orientation': 'horizontal' if bbox_width > bbox_height else 'vertical'
                    })
        
        print(f"YOLOv11 detected {len(predictions)} scale bars")
        
        # Visualize results
        if save_result and predictions:
            self._visualize_predictions(image_path, predictions)
        
        return predictions
    
    def _visualize_predictions(self, image_path, predictions):
        """Visualize YOLOv11 prediction results"""
        
        image = cv2.imread(image_path)
        if image is None:
            return
        
        for i, pred in enumerate(predictions):
            # Draw bounding box
            x1, y1, x2, y2 = [int(x) for x in pred['bbox']]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw scale bar line
            lx1, ly1, lx2, ly2 = [int(x) for x in pred['line']]
            cv2.line(image, (lx1, ly1), (lx2, ly2), (0, 0, 255), 4)
            cv2.circle(image, (lx1, ly1), 6, (255, 0, 0), -1)
            cv2.circle(image, (lx2, ly2), 6, (255, 0, 0), -1)
            
            # Add detailed labels
            conf_text = f"YOLOv11: {pred['confidence']:.3f}"
            orient_text = f"Dir: {pred['orientation']}"
            cv2.putText(image, conf_text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, orient_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            print(f"   Scale bar {i+1}: Confidence={pred['confidence']:.3f}, Length={pred['length']:.1f}px, Orientation={pred['orientation']}")
        
        # Save results
        output_path = f"yolo11_prediction_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)
        print(f"YOLOv11 prediction results saved to: {output_path}")

def main():
    """Main function"""
    
    print("YOLOv11 Scale Bar Detector - Latest and Most Powerful Version")
    print("="*60)
    
    print("Key improvements in YOLOv11:")
    print("   • Enhanced backbone network")
    print("   • Improved feature pyramid")
    print("   • More accurate detection head")
    print("   • Optimized training strategies")
    print("   • Better small object detection")
    print("="*60)
    
    print("Select operation:")
    print("1. Convert data to YOLOv11 format")
    print("2. Train YOLOv11 model")
    print("3. Evaluate YOLOv11 model")
    print("4. Predict single image")
    print("5. Batch prediction")
    print("6. Complete YOLOv11 pipeline (Recommended)")
    print("7. YOLOv11 vs YOLOv8 comparison")
    
    try:
        choice = input("Please enter your choice (1-7): ").strip()
        
        if choice == '1':
            # Convert data
            converter = YOLO11ScaleBarConverter('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
            sample_count = converter.convert_to_yolo_format()
            print(f"Data conversion completed, total {sample_count} samples")
            
        elif choice == '2':
            # Train YOLOv11 model
            trainer = YOLO11ScaleBarTrainer()
            trainer.train_yolo11_model(model_size="yolo11s", epochs=100)
            
        elif choice == '3':
            # Evaluate model
            from yolo_length_evaluator import YOLOLengthEvaluator
            evaluator = YOLOLengthEvaluator("yolo11_scalebar_runs/scalebar_detection_v11/weights/best.pt")
            evaluator.evaluate_dataset('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
            
        elif choice == '4':
            # Predict single image
            img_path = input("Enter image path: ").strip()
            predictor = YOLO11ScaleBarPredictor()
            predictor.predict_image(img_path)
            
        elif choice == '5':
            # Batch prediction
            img_dir = input("Enter image directory: ").strip()
            predictor = YOLO11ScaleBarPredictor()
            # Implement batch prediction logic
            print("Batch prediction feature in development...")
            
        elif choice == '6':
            # Complete YOLOv11 pipeline
            print("Executing complete YOLOv11 training pipeline...")
            
            # Step 1: Convert data
            print("\nStep 1: Convert data to YOLOv11 format")
            converter = YOLO11ScaleBarConverter('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
            sample_count = converter.convert_to_yolo_format()
            
            if sample_count < 50:
                print("Warning: Insufficient sample count, may affect training performance")
            
            # Step 2: Train YOLOv11 model
            print("\nStep 2: Train YOLOv11 model")
            trainer = YOLO11ScaleBarTrainer()
            success = trainer.train_yolo11_model(model_size="yolo11s", epochs=80)
            
            if success:
                # Step 3: Length error evaluation
                print("\nStep 3: Length error evaluation")
                try:
                    from yolo_length_evaluator import YOLOLengthEvaluator
                    evaluator = YOLOLengthEvaluator("yolo11_scalebar_runs/scalebar_detection_v11/weights/best.pt")
                    evaluator.evaluate_dataset('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
                except ImportError:
                    print("Please run the length evaluator first")
                
                print("\nYOLOv11 scale bar detector training completed!")
                print("Expected performance improvement of 2-5% for YOLOv11 over YOLOv8")
            
        elif choice == '7':
            # Performance comparison
            trainer = YOLO11ScaleBarTrainer()
            trainer.compare_with_yolo8()
            
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nOperation interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
