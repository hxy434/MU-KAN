#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Analysis Demonstration Script - Integrated with UKAN Model
Demonstrates how to use particle analysis functionality
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from particle_analyzer import ParticleAnalyzer
from scale_detector import ScaleDetectorWithPreprocessing
from mock_ocr import process as ocr_process
from improved_ocr_guided_detector import ImprovedOCRGuidedDetector
import time
import glob
import sys
import torch
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize
from PIL import Image
import random
from datetime import datetime

# Add seg-MOGA path to system path

import archs
from dataset import Dataset

ACCESS_TOKEN = "24.a6279612cc9567f9aca9316df9294369.2592000.1767946592.282335-119495206"

# Create output folder
def create_output_directory():
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"particle_analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f" Results will be saved to: {output_dir}")
    return output_dir

# Global output directory
OUTPUT_DIR = create_output_directory()

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_ukan_model():
    """Load UKAN model"""
    # Load configuration file
    config_path = 'D:/python-learn/K/seg-MOGA/outputs/lizi_UKAN/config.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    
    # Create model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        use_Moga=config['use_Moga']
    )
    
    # Load weights
    ckpt = torch.load('D:/python-learn/K/seg-MOGA/outputs/lizi_UKAN/model.pth', map_location='cpu')
    try:
        model.load_state_dict(ckpt)
    except Exception as e:
        print("Exception occurred while loading weights, using non-strict mode:")
        print("Exception:", e)
        model.load_state_dict(ckpt, strict=False)
    
    model.eval()
    return model, config

def preprocess_image_for_ukan(image, config):
    """Use exactly the same preprocessing as during training"""
    # Convert to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use exactly the same preprocessing as during validation
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),  # Consistent with transforms.Normalize() in val.py
    ])
    
    # Apply transformations
    augmented = val_transform(image=img)
    img = augmented['image']
    
    # Convert to tensor
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
    
    return img

def ukan_inference(image, model, config):
    """Perform inference using UKAN model"""
    # Preprocessing
    input_tensor = preprocess_image_for_ukan(image, config)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]  # Take first batch, first channel
    
    # Binarization (consistent with val.py)
    pred_mask = (prob_map >= 0.5).astype(np.uint8)
    
    # Resize back to original image dimensions
    original_h, original_w = image.shape[:2]
    pred_mask_resized = cv2.resize(pred_mask, (original_w, original_h))
    
    return prob_map, pred_mask_resized

def extract_scale_length_from_ocr(image_path):
    result = ocr_process(image_path)
    print("Raw OCR result:", result)
    words_result = result.get("words_result", [])
    for item in words_result:
        words = item["words"].replace(' ', '')
        print(f"Processing OCR text: '{words}'")
        for unit in ["nm", "μm", "um"]:
            if unit in words:
                num_str = words.replace(unit, "")
                try:
                    value = float(num_str)
                    print(f"Found scale bar: {value} {unit} (original recognition)")
                    return value, unit
                except:
                    continue
    print("No scale bar information found")
    return None, None

def detect_particles_with_yolo(image, model_path='D:/python-learn/K/seg-MOGA/runs/segment/kan/weights/best.pt'):
    """Detect particles using YOLO model"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(image, conf=0.3, iou=0.5)
        return results[0]  # Return results for first image
    except Exception as e:
        print(f"YOLO detection failed: {e}")
        return None

def calculate_particle_features_from_mask(mask, pixel_to_real_ratio=None, unit="μm"):
    """Calculate particle features from mask"""
    from scipy import ndimage
    
    # Connected component analysis
    labeled_mask, num_features = ndimage.label(mask > 0)
    
    particle_features = []
    
    for i in range(1, num_features + 1):
        # Extract single connected component
        single_mask = (labeled_mask == i).astype(np.uint8)
        
        # Calculate area (pixel count)
        area_pixels = np.sum(single_mask)
        
        # Calculate equivalent diameter
        diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        
        # Convert to physical units
        if pixel_to_real_ratio:
            area_real = area_pixels * (pixel_to_real_ratio ** 2)
            diameter_real = diameter_pixels * pixel_to_real_ratio
        else:
            area_real = area_pixels
            diameter_real = diameter_pixels
        
        # Calculate perimeter
        contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0
        
        # Calculate circularity
        circularity = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0
        
        particle_features.append({
            'id': i,
            'area_pixels': area_pixels,
            'area_real': area_real,
            'diameter_pixels': diameter_pixels,
            'diameter_real': diameter_real,
            'perimeter': perimeter,
            'circularity': circularity
        })
    
    return particle_features, labeled_mask

def calculate_particle_features(masks, pixel_to_real_ratio=None, unit="μm"):
    """Calculate particle features"""
    particle_features = []
    
    for i, mask in enumerate(masks):
        # Calculate area (pixel count)
        area_pixels = np.sum(mask)
        
        # Calculate equivalent diameter
        diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        
        # Convert to physical units
        if pixel_to_real_ratio:
            area_real = area_pixels * (pixel_to_real_ratio ** 2)
            diameter_real = diameter_pixels * pixel_to_real_ratio
        else:
            area_real = area_pixels
            diameter_real = diameter_pixels
        
        # Calculate perimeter
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0
        
        # Calculate circularity
        circularity = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0
        
        particle_features.append({
            'id': i,
            'area_pixels': area_pixels,
            'area_real': area_real,
            'diameter_pixels': diameter_pixels,
            'diameter_real': diameter_real,
            'perimeter': perimeter,
            'circularity': circularity
        })
    
    return particle_features

def plot_particle_distributions(particle_features, pixel_to_real_ratio=None, unit="μm", save_path=None):
    """Plot particle distribution graphs"""
    if not particle_features:
        print("No particles detected, cannot plot distribution graphs")
        return
    
    # Extract data
    areas = [p['area_real'] for p in particle_features]
    diameters = [p['diameter_real'] for p in particle_features]
    circularities = [p['circularity'] for p in particle_features]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Particle Distribution Analysis', fontsize=24, fontweight='bold')
    
    # 1. Area distribution histogram
    axes[0, 0].hist(areas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel(f'Particle Area ({unit}$^2$)', fontsize=20)
    axes[0, 0].set_ylabel('Frequency', fontsize=20)
    axes[0, 0].set_title('Particle Area Distribution', fontsize=22)
    axes[0, 0].tick_params(axis='both', labelsize=18)
    
    # 2. Diameter distribution histogram
    axes[0, 1].hist(diameters, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel(f'Particle Diameter ({unit})', fontsize=20)
    axes[0, 1].set_ylabel('Frequency', fontsize=20)
    axes[0, 1].set_title('Particle Diameter Distribution', fontsize=22)
    axes[0, 1].tick_params(axis='both', labelsize=18)
    
    # 3. Circularity distribution histogram
    axes[1, 0].hist(circularities, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Circularity', fontsize=20)
    axes[1, 0].set_ylabel('Frequency', fontsize=20)
    axes[1, 0].set_title('Particle Circularity Distribution', fontsize=22)
    axes[1, 0].tick_params(axis='both', labelsize=18)
    
    # 4. Area vs diameter scatter plot
    axes[1, 1].scatter(diameters, areas, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel(f'Particle Diameter ({unit})', fontsize=20)
    axes[1, 1].set_ylabel(f'Particle Area ({unit}$^2$)', fontsize=20)
    axes[1, 1].set_title('Area vs Diameter Relationship', fontsize=22)
    axes[1, 1].tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribution graph saved: {save_path}")
    plt.close()

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALE_DETECTOR_PATH = os.path.join(BASE_DIR, 'scale_detector_kan.pth')

def analyze_sem_image(image_path):
    print(f"\n Analyzing SEM image: {image_path}")
    print("=" * 40)
    if not os.path.exists(image_path):
        print(f" Image file does not exist: {image_path}")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f" Cannot read image: {image_path}")
        return
    print(f"Image dimensions: {image.shape}")
    
    # 1. Detect scale bar using improved OCR-guided detector
    coords, scale_length_pixels, confidence, scale_length_real, unit = detect_scale_with_improved_detector(image_path)
    
    # Display detailed coordinate information
    if coords is not None:
        print(f"   Scale bar coordinate information:")
        print(f"   Detection confidence: {confidence:.3f}")
        print(f"   Scale bar start coordinates: ({coords[0]}, {coords[1]})")
        print(f"   Scale bar end coordinates: ({coords[2]}, {coords[3]})")
        print(f"   Scale bar pixel length: {scale_length_pixels:.1f} pixels")
        
        # Calculate scale bar angle
        dx = coords[2] - coords[0]
        dy = coords[3] - coords[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"   Scale bar angle: {angle:.1f}°")
    else:
        print(" No scale bar detected")
        # If improved detector fails, try original detector as fallback
        print(" Trying original detector as fallback...")
        scale_detector = ScaleDetectorWithPreprocessing(
            model_path=SCALE_DETECTOR_PATH,
            input_size=(320, 320),
            hidden_dim=64
        )
        coords, scale_length_pixels, confidence = scale_detector.detect_scale(image)
        scale_length_real, unit = extract_scale_length_from_ocr(image_path)
    
    # 2. Automatic conversion
    if scale_length_pixels and scale_length_real:
        pixel_to_real_ratio = scale_length_real / scale_length_pixels
        print(f"Physical length per pixel: {pixel_to_real_ratio} {unit}")
    else:
        pixel_to_real_ratio = None
        unit = "None"  # No scale bar detected
        print("Length per pixel: Unknown (pixel units only)")
    
    # 3. Initialize particle analyzer (for fallback option)
    scale_detector = ScaleDetectorWithPreprocessing(
        model_path=SCALE_DETECTOR_PATH,
        input_size=(320, 320),
        hidden_dim=64
    )
    analyzer = ParticleAnalyzer(scale_detector)
    
    # 4. Detect particles using UKAN model
    print("\n Detecting particles using UKAN model...")
    try:
        # Load UKAN model
        ukan_model, ukan_config = load_ukan_model()
        
        # UKAN inference
        prob_map, pred_mask = ukan_inference(image, ukan_model, ukan_config)
        
        # Statistical information
        total_pixels = pred_mask.size
        positive_pixels = np.sum(pred_mask > 0)
        positive_ratio = positive_pixels / total_pixels * 100
        
        print(f"UKAN detection results:")
        print(f"   Total pixels: {total_pixels}")
        print(f"   Positive pixels: {positive_pixels}")
        print(f"   Positive pixel ratio: {positive_ratio:.4f}%")
        
        # Calculate particle features
        particle_features, labeled_mask = calculate_particle_features_from_mask(
            pred_mask,
            pixel_to_real_ratio,
            unit if unit else "μm"
        )
        
        if particle_features:
            print(f"   Number of particles detected: {len(particle_features)}")
            
            # 6. Plot distribution graphs
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
            plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
            
            # 7. Output statistical information
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            
            print(f"\n UKAN particle detection results:")
            print(f"   Number of particles detected: {len(particle_features)}")
            print(f"   Average area: {np.mean(areas):.2f} {unit}²")
            print(f"   Average diameter: {np.mean(diameters):.2f} {unit}")
            print(f"   Average circularity: {np.mean(circularities):.3f}")
            print(f"   Area standard deviation: {np.std(areas):.2f} {unit}²")
            print(f"   Diameter standard deviation: {np.std(diameters):.2f} {unit}")
            
            # Save visualization results
            save_visualization_results(image, prob_map, pred_mask, base_name, particle_features, labeled_mask)
            
            # Save analysis report
            save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "MU-KAN", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
            
        else:
            print(" No particles detected by UKAN")
            # Save analysis report (even if no particles detected)
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "MU-KAN", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
            
    except Exception as e:
        print(f" UKAN detection failed: {e}")
        print("Trying YOLO detection...")
        
        # Use YOLO as fallback
        yolo_results = detect_particles_with_yolo(image)
        
        if yolo_results:
            print(f"YOLO detected {len(yolo_results.masks)} particles")
            
            # Calculate particle features
            particle_features = calculate_particle_features(
                yolo_results.masks.data.cpu().numpy(),
                pixel_to_real_ratio,
                unit if unit else "μm"
            )
            
            # Plot distribution graphs
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
            plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
            
            # Save YOLO masks
            masks = yolo_results.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_mask_{i+1}.png")
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_uint8)
                print(f" YOLO mask {i+1} saved: {mask_path}")
            
            # Save combined mask
            combined_mask = np.max(masks, axis=0).astype(np.uint8)
            combined_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_combined_mask.png")
            combined_mask_uint8 = (combined_mask * 255).astype(np.uint8)
            cv2.imwrite(combined_mask_path, combined_mask_uint8)
            print(f" YOLO combined mask saved: {combined_mask_path}")
            
            # Output statistical information
            if particle_features:
                areas = [p['area_real'] for p in particle_features]
                diameters = [p['diameter_real'] for p in particle_features]
                circularities = [p['circularity'] for p in particle_features]
                
                print(f"\n YOLO particle detection results:")
                print(f"   Number of particles detected: {len(particle_features)}")
                print(f"   Average area: {np.mean(areas):.2f} {unit}²")
                print(f"   Average diameter: {np.mean(diameters):.2f} {unit}")
                print(f"   Average circularity: {np.mean(circularities):.3f}")
                print(f"   Area standard deviation: {np.std(areas):.2f} {unit}²")
                print(f"   Diameter standard deviation: {np.std(diameters):.2f} {unit}")
                
                # Save analysis report
                save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "YOLO", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
        else:
            print(" No particles detected by YOLO either, using traditional method...")
            # Use original analysis method as fallback
            results = analyzer.analyze_image_with_scale(
                image,
                scale_length_real=scale_length_real,
                unit=unit if unit else "μm",
                particle_method="watershed",
                save_path=os.path.join(OUTPUT_DIR, f"sem_analysis_{os.path.basename(image_path)}.jpg")
            )
            # Output analysis results
            if results:
                print(f"\n Traditional method analysis results:")
                print(f"   Scale bar length: {results['scale_info']['length_pixels']:.1f} pixels")
                print(f"   Number of particles detected: {results['analysis']['count']}")
                print(f"   Average diameter: {results['analysis']['diameter_stats']['mean']:.2f}{results['scale_info']['unit']}")
                print(f"   Average area: {results['analysis']['area_stats']['mean']:.2f}{results['scale_info']['unit']}²")
                print(f"   Average circularity: {results['analysis']['circularity_stats']['mean']:.3f}")
                
                # Save analysis report
                save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "Traditional Method", {"scale_length_pixels": scale_length_pixels, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
    
    print("\n Analysis completed!")

# In analyze_sem_image_ukan_only and analyze_sem_image_yolo_only, construct scale_info and return it

def analyze_sem_image_ukan_only(image_path):
    """Analyze SEM image using UKAN model only"""
    print(f"\n Analyzing SEM image with UKAN model: {image_path}")
    print("=" * 40)
    if not os.path.exists(image_path):
        print(f" Image file does not exist: {image_path}")
        return
    scale_info = {}
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f" Cannot read image: {image_path}")
        return
    print(f"Image dimensions: {image.shape}")
    
    # 1. Detect scale bar using improved OCR-guided detector
    coords, scale_length_pixels, confidence, scale_length_real, unit = detect_scale_with_improved_detector(image_path)
    
    # Display detailed coordinate information
    if coords is not None:
        print(f" Scale bar coordinate information:")
        print(f"   Detection confidence: {confidence:.3f}")
        print(f"   Scale bar start coordinates: ({coords[0]}, {coords[1]})")
        print(f"   Scale bar end coordinates: ({coords[2]}, {coords[3]})")
        print(f"   Scale bar pixel length: {scale_length_pixels:.1f} pixels")
        
        # Calculate scale bar angle
        dx = coords[2] - coords[0]
        dy = coords[3] - coords[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"   Scale bar angle: {angle:.1f}°")
    else:
        print(" No scale bar detected")
        # If improved detector fails, try original detector as fallback
        print(" Trying original detector as fallback...")
        scale_detector = ScaleDetectorWithPreprocessing(
            model_path=SCALE_DETECTOR_PATH,
            input_size=(320, 320),
            hidden_dim=64
        )
        coords, scale_length_pixels, confidence = scale_detector.detect_scale(image)
        scale_length_real, unit = extract_scale_length_from_ocr(image_path)
    
    print(f"Raw OCR recognition result: {scale_length_real} {unit}")
    
    # 2. Automatic conversion
    if scale_length_pixels and scale_length_real:
        pixel_to_real_ratio = scale_length_real / scale_length_pixels
        print(f"Physical length per pixel: {pixel_to_real_ratio} {unit}")
        scale_info = {
            "scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None,
            "scale_length_real": scale_length_real,
            "unit": unit,
            "pixel_to_real_ratio": float(pixel_to_real_ratio) if pixel_to_real_ratio is not None else None,
            "coords": coords if coords is not None else None
        }
    else:
        pixel_to_real_ratio = None
        unit = "None"  # No scale bar detected
        print("Length per pixel: Unknown (pixel units only)")
        scale_info = {}
    # 5. Detect particles using UKAN model
    print("\n Detecting particles using UKAN model...")
    try:
        ukan_model, ukan_config = load_ukan_model()
        prob_map, pred_mask = ukan_inference(image, ukan_model, ukan_config)
        total_pixels = pred_mask.size
        positive_pixels = np.sum(pred_mask > 0)
        positive_ratio = positive_pixels / total_pixels * 100
        print(f"UKAN detection results:")
        print(f"   Total pixels: {total_pixels}")
        print(f"   Positive pixels: {positive_pixels}")
        print(f"   Positive pixel ratio: {positive_ratio:.4f}%")
        particle_features, labeled_mask = calculate_particle_features_from_mask(
            pred_mask,
            pixel_to_real_ratio,
            unit if unit else "μm"
        )
        if particle_features:
            print(f"   Number of particles detected: {len(particle_features)}")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
            plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            print(f"\n UKAN particle detection results:")
            print(f"   Number of particles detected: {len(particle_features)}")
            print(f"   Average area: {np.mean(areas):.2f} {unit}²")
            print(f"   Average diameter: {np.mean(diameters):.2f} {unit}")
            print(f"   Average circularity: {np.mean(circularities):.3f}")
            print(f"   Area standard deviation: {np.std(areas):.2f} {unit}²")
            print(f"   Diameter standard deviation: {np.std(diameters):.2f} {unit}")
            save_visualization_results(image, prob_map, pred_mask, base_name, particle_features, labeled_mask)
            save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "MU-KAN Segmentation and YOLOv11 Detection", scale_info)
        else:
            print(" No particles detected by UKAN")
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "MU-KAN", scale_info)
        print("\n Analysis completed!")
        print(f"Analysis results directory: {OUTPUT_DIR}")
        return OUTPUT_DIR, {"scale_info": scale_info}
    except Exception as e:
        print(f" UKAN detection failed: {e}")
        print("UKAN detection failed, please check model files or try other methods")
        print("\n Analysis completed!")
        print(f"Analysis results directory: {OUTPUT_DIR}")
        return OUTPUT_DIR, {"scale_info": scale_info}

def analyze_sem_image_yolo_only(image_path):
    """Analyze SEM image using YOLO model only"""
    print(f"\n Analyzing SEM image with YOLO model: {image_path}")
    print("=" * 40)
    if not os.path.exists(image_path):
        print(f" Image file does not exist: {image_path}")
        return
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f" Cannot read image: {image_path}")
        return
    print(f"Image dimensions: {image.shape}")
    
    # 1. Detect scale bar using improved OCR-guided detector
    coords, scale_length_pixels, confidence, scale_length_real, unit = detect_scale_with_improved_detector(image_path)
    
    # Display detailed coordinate information
    if coords is not None:
        print(f" Scale bar coordinate information:")
        print(f"   Detection confidence: {confidence:.3f}")
        print(f"   Scale bar start coordinates: ({coords[0]}, {coords[1]})")
        print(f"   Scale bar end coordinates: ({coords[2]}, {coords[3]})")
        print(f"   Scale bar pixel length: {scale_length_pixels:.1f} pixels")
        
        # Calculate scale bar angle
        dx = coords[2] - coords[0]
        dy = coords[3] - coords[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"   Scale bar angle: {angle:.1f}°")
    else:
        print(" No scale bar detected")
        # If improved detector fails, try original detector as fallback
        print(" Trying original detector as fallback...")
        scale_detector = ScaleDetectorWithPreprocessing(
            model_path=SCALE_DETECTOR_PATH,
            input_size=(320, 320),
            hidden_dim=64
        )
        coords, scale_length_pixels, confidence = scale_detector.detect_scale(image)
        scale_length_real, unit = extract_scale_length_from_ocr(image_path)
    
    print(f"Raw OCR recognition result: {scale_length_real} {unit}")
    
    # 2. Automatic conversion
    if scale_length_pixels and scale_length_real:
        pixel_to_real_ratio = scale_length_real / scale_length_pixels
        print(f"Physical length per pixel: {pixel_to_real_ratio} {unit}")
    else:
        pixel_to_real_ratio = None
        unit = "None"  # No scale bar detected
        print("Length per pixel: Unknown (pixel units only)")
    
    # 5. Detect particles using YOLO
    print("\n Detecting particles using YOLO...")
    yolo_results = detect_particles_with_yolo(image)
    
    if yolo_results:
        print(f"YOLO detected {len(yolo_results.masks)} particles")
        
        # Calculate particle features
        particle_features = calculate_particle_features(
            yolo_results.masks.data.cpu().numpy(),
            pixel_to_real_ratio,
            unit if unit else "μm"
        )
        
        # Plot distribution graphs
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
        plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
        
        # Save YOLO masks
        masks = yolo_results.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_mask_{i+1}.png")
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)
            print(f" YOLO mask {i+1} saved: {mask_path}")
        
        # Save combined mask
        combined_mask = np.max(masks, axis=0).astype(np.uint8)
        combined_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_combined_mask.png")
        combined_mask_uint8 = (combined_mask * 255).astype(np.uint8)
        cv2.imwrite(combined_mask_path, combined_mask_uint8)
        print(f" YOLO combined mask saved: {combined_mask_path}")
        
        # Output statistical information
        if particle_features:
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            
            print(f"\n YOLO particle detection results:")
            print(f"   Number of particles detected: {len(particle_features)}")
            print(f"   Average area: {np.mean(areas):.2f} {unit}²")
            print(f"   Average diameter: {np.mean(diameters):.2f} {unit}")
            print(f"   Average circularity: {np.mean(circularities):.3f}")
            print(f"   Area standard deviation: {np.std(areas):.2f} {unit}²")
            print(f"   Diameter standard deviation: {np.std(diameters):.2f} {unit}")
            
            # Save analysis report
            save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "YOLO", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
        else:
            print(" No particles detected by YOLO")
            # Save analysis report (even if no particles detected)
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "YOLO", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
    else:
        print(" YOLO detection failed, using traditional method...")
        # Use original analysis method as fallback
        results = analyzer.analyze_image_with_scale(
            image,
            scale_length_real=scale_length_real,
            unit=unit if unit else "μm",
            particle_method="watershed",
            save_path=os.path.join(OUTPUT_DIR, f"sem_analysis_{os.path.basename(image_path)}.jpg")
        )
        # Output analysis results
        if results:
            print(f"\n Traditional method analysis results:")
            print(f"   Scale bar length: {results['scale_info']['length_pixels']:.1f} pixels")
            print(f"   Number of particles detected: {results['analysis']['count']}")
            print(f"   Average diameter: {results['analysis']['diameter_stats']['mean']:.2f}{results['scale_info']['unit']}")
            print(f"   Average area: {results['analysis']['area_stats']['mean']:.2f}{results['scale_info']['unit']}²")
            print(f"   Average circularity: {results['analysis']['circularity_stats']['mean']:.3f}")
                
            # Save analysis report
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "Traditional Method", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
    
    print("\n Analysis completed!")
    print(f"Analysis results directory: {OUTPUT_DIR}")
    return OUTPUT_DIR

def detect_scale_with_improved_detector(image_path, access_token=ACCESS_TOKEN):
    """
    Detect scale bar using improved OCR-guided detector
    Returns: (coords, scale_length_pixels, confidence, scale_length_real, unit)
    """
    print(f" Detecting scale bar using improved OCR-guided detector...")
    
    # Initialize improved detector
    model_path = "../epoch80.pt"  # Weight file in root directory, need to go up one level when running from backend directory
    try:
        detector = ImprovedOCRGuidedDetector(model_path, access_token)
    except Exception as e:
        print(f" Failed to initialize improved detector: {e}")
        return None, None, None, None, None
    
    # Perform hybrid detection
    try:
        detection_result = detector.detect_scale_hybrid(image_path, conf=0.2)
        
        if not detection_result['final_results']:
            print(" No scale bar detected by improved detector")
            return None, None, None, None, None
        
        # Get best detection result
        best_result = detection_result['final_results'][0]
        bbox = best_result['bbox']
        
        # Calculate scale bar coordinates and length
        x1, y1, x2, y2 = bbox
        coords = [int(x1), int(y1), int(x2), int(y2)]
        scale_length_pixels = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        confidence = float(best_result.get('confidence', 0.5))
        
        # Extract physical length and unit from OCR results
        scale_length_real = None
        unit = None
        
        # Try to extract scale bar information from text regions
        for region in detection_result['text_regions']:
            text = region['text']
            # Extract numbers and units using regular expression
            import re
            pattern = r'(\d+(?:\.\d+)?)\s*(nm|μm|um|mm|cm|m)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scale_length_real = float(match.group(1))
                unit = match.group(2)
                break
        
        # If not extracted by OCR, try original OCR function
        if scale_length_real is None:
            scale_length_real, unit = extract_scale_length_from_ocr(image_path)
        
        print(f"  Improved detector detection successful:")
        print(f"   Scale bar coordinates: ({coords[0]}, {coords[1]}) -> ({coords[2]}, {coords[3]})")
        print(f"   Pixel length: {scale_length_pixels:.1f}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Physical length: {scale_length_real} {unit}")
        
        # Save visualization results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(OUTPUT_DIR, f"{base_name}_improved_detection.png")
        detector.visualize_results(detection_result, vis_path)
        
        return coords, scale_length_pixels, confidence, scale_length_real, unit
        
    except Exception as e:
        print(f" Improved detector detection failed: {e}")
        return None, None, None, None, None

def save_visualization_results(image, prob_map, pred_mask, base_name, particle_features=None, labeled_mask=None):
    """Save visualization results and annotate particle numbers and segmentation boundaries on original image"""
    # Use global output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save binary mask
    mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_binary_mask.png")
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_uint8)
    print(f" Binary mask saved: {mask_path}")
    
    # Save probability map
    prob_path = os.path.join(OUTPUT_DIR, f"{base_name}_probability_map.png")
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(prob_path, prob_uint8)
    print(f" Probability map saved: {prob_path}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'MU-KAN Model Analysis Results - {base_name}', fontsize=16, fontweight='bold')
    
    # 1. Original image
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. Probability map
    axes[0, 1].imshow(prob_map, cmap='hot')
    axes[0, 1].set_title('Probability Map')
    axes[0, 1].axis('off')
    
    # 3. Binarized mask
    axes[1, 0].imshow(pred_mask, cmap='gray')
    axes[1, 0].set_title('Binary Mask (Threshold 0.5)')
    axes[1, 0].axis('off')
    
    # 4. Overlay display
    overlay = original_rgb.copy()
    mask_resized = cv2.resize(pred_mask, (original_rgb.shape[1], original_rgb.shape[0]))
    overlay[mask_resized > 0] = [0, 255, 0]  # Green for detection area
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay Display (Green=Detection Area)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save results
    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Visualization results saved: {save_path}")
    plt.close()

    # Save original image with numbered particles
    if particle_features is not None and labeled_mask is not None and len(particle_features) > 0:
        numbered_img = image.copy()
        for idx, feat in enumerate(particle_features, 1):
            mask = (labeled_mask == idx).astype(np.uint8)
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(numbered_img, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        numbered_path = os.path.join(OUTPUT_DIR, f"{base_name}_numbered.png")
        cv2.imwrite(numbered_path, numbered_img)
        print(f" Numbered original image saved: {numbered_path}")
    # Save original image with segmentation boundaries overlay
    contour_img = image.copy()
    contours, _ = cv2.findContours((pred_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)  # Red lines
    contour_path = os.path.join(OUTPUT_DIR, f"{base_name}_contour.png")
    cv2.imwrite(contour_path, contour_img)
    print(f" Segmentation boundary overlay image saved: {contour_path}")

def save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, method="MU-KAN", scale_info=None):
    report_path = os.path.join(OUTPUT_DIR, f"analysis_report_{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Particle Analysis Report\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Scale Unit: {unit}\n")
        if scale_info:
            f.write(f"Scale Length (pixels): {scale_info.get('scale_length_pixels', 'Not detected')}\n")
            f.write(f"OCR Detected Physical Length: {scale_info.get('scale_length_real', 'Not detected')} {scale_info.get('unit', '')}\n")
            # Add coordinate information to report
            if 'coords' in scale_info:
                coords = scale_info['coords']
                f.write(f"Scale Start Coordinates: ({coords[0]}, {coords[1]})\n")
                f.write(f"Scale End Coordinates: ({coords[2]}, {coords[3]})\n")
                dx = coords[2] - coords[0]
                dy = coords[3] - coords[1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                f.write(f"Scale Angle: {angle:.1f}°\n")
        f.write(f"Pixel to Physical Unit Ratio: {pixel_to_real_ratio}\n\n")
        if particle_features:
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            f.write(f"Detection Results Statistics:\n")
            f.write(f"  Number of Particles Detected: {len(particle_features)}\n")
            f.write(f"  Average Area: {np.mean(areas):.2f} {unit}²\n")
            f.write(f"  Average Diameter: {np.mean(diameters):.2f} {unit}\n")
            f.write(f"  Average Circularity: {np.mean(circularities):.3f}\n")
            f.write(f"  Area Standard Deviation: {np.std(areas):.2f} {unit}²\n")
            f.write(f"  Diameter Standard Deviation: {np.std(diameters):.2f} {unit}\n\n")
            f.write(f"Detailed Particle Information:\n")
            for i, particle in enumerate(particle_features):
                f.write(f"  Particle {i+1}:\n")
                f.write(f"    Area: {particle['area_real']:.2f} {unit}²\n")
                f.write(f"    Diameter: {particle['diameter_real']:.2f} {unit}\n")
                f.write(f"    Circularity: {particle['circularity']:.3f}\n")
        else:
            f.write("No particles detected\n")
    print(f" Analysis report saved: {report_path}")

if __name__ == "__main__":
    print(" Fully Automatic KAN Scale Bar + Particle Detection + Distribution Analysis (Batch Mode)")
    print("=" * 50)
    
    # Select detection method
    print("Please select particle detection method:")
    print("1 = UKAN Semantic Segmentation (Recommended, better performance)")
    print("2 = YOLO Object Detection (Fallback option)")
    print("3 = Auto Select (UKAN first, YOLO if failed)")
    
    mode = input("Please enter your selection (1/2/3): ").strip()
    
    # Support batch processing of all images in folder
    folder = input("Please enter SEM image folder path (e.g., datasets/lizi/images or single image path): ").strip()
    
    # Remove possible quotation marks
    folder = folder.strip('"\'')
    
    if os.path.isdir(folder):
        # Process all png/jpg images
        image_list = sorted(glob.glob(folder + "/*.png") + glob.glob(folder + "/*.jpg"))
        print(f"Total {len(image_list)} images detected, will analyze sequentially...")
        for image_path in image_list:
            if mode == "1":
                analyze_sem_image_ukan_only(image_path)
            elif mode == "2":
                analyze_sem_image_yolo_only(image_path)
            elif mode == "3":
                analyze_sem_image(image_path)
            else:
                print("Invalid selection, using auto mode")
                analyze_sem_image(image_path)
            time.sleep(0.7)  # Prevent Baidu OCR QPS limit exceeded
    else:
        # Single image
        if mode == "1":
            analyze_sem_image_ukan_only(folder)
        elif mode == "2":
            analyze_sem_image_yolo_only(folder)
        elif mode == "3":
            analyze_sem_image(folder)
        else:
            print("Invalid selection, using auto mode")
            analyze_sem_image(folder)
