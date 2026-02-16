#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-Guided Scale Bar Detector
First recognize scale bar numbers and units, then expand search for scale bar centered on this region
"""

import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import re
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import mock_ocr

class OCRGuidedScaleDetector:
    def __init__(self, model_path, access_token=None):
        """
        Initialize OCR-guided scale bar detector
        
        Args:
            model_path: Path to YOLO model
            access_token: Baidu OCR access_token
        """
        self.model = YOLO(model_path)
        self.access_token = access_token
        
        # Scale bar unit patterns
        self.scale_units = ['um', 'μm', 'nm', 'mm', 'cm', 'm', 'pixel', 'px']
        self.scale_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(' + '|'.join(self.scale_units) + ')', re.IGNORECASE)
        
        # Expansion search parameters
        self.expansion_ratio = 3.0  # Expansion multiplier
        self.min_confidence = 0.1   # Minimum confidence for expanded search
        
        print(" OCR-guided scale bar detector initialized successfully")
        
    def detect_scale_text(self, image_path):
        """
        Detect scale bar text in image
        
        Args:
            image_path: Path to image file
            
        Returns:
            list: List of detected text regions, each containing text, location, and confidence
        """
        print(f" Detecting text with OCR: {os.path.basename(image_path)}")
        
        # Perform text detection using mock_ocr
        ocr_result = mock_ocr.process(image_path, self.access_token)
        
        if ocr_result.get("error_code", 0) != 0:
            print(f" OCR detection failed: {ocr_result.get('error_msg', 'Unknown error')}")
            return []
        
        text_regions = []
        words_result = ocr_result.get("words_result", [])
        
        for item in words_result:
            text = item.get("words", "")
            location = item.get("location", {})
            
            # Check if text matches scale bar pattern
            if self.scale_pattern.search(text):
                # Ensure complete location information
                if all(key in location for key in ['left', 'top', 'width', 'height']):
                    text_regions.append({
                        'text': text,
                        'location': location,
                        'confidence': 0.9  # OCR confidence
                    })
                    print(f"    Detected scale bar text: {text} Location: {location}")
                else:
                    print(f"    Incomplete location info for scale bar text: {text} Location: {location}")
        
        print(f"   Total scale bar texts detected: {len(text_regions)}")
        return text_regions
    
    def create_search_regions(self, image_path, text_regions):
        """
        Create search regions based on detected text
        
        Args:
            image_path: Path to image file
            text_regions: List of detected text regions
            
        Returns:
            list: List of expanded search regions
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        height, width = image.shape[:2]
        search_regions = []
        
        for region in text_regions:
            location = region['location']
            text = region['text']
            
            # Get text region coordinates
            x1 = location['left']
            y1 = location['top']
            x2 = x1 + location['width']
            y2 = y1 + location['height']
            
            # Calculate expanded region
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Expand search region
            expanded_width = int(location['width'] * self.expansion_ratio)
            expanded_height = int(location['height'] * self.expansion_ratio)
            
            # Ensure coordinates stay within image boundaries
            new_x1 = max(0, center_x - expanded_width // 2)
            new_y1 = max(0, center_y - expanded_height // 2)
            new_x2 = min(width, center_x + expanded_width // 2)
            new_y2 = min(height, center_y + expanded_height // 2)
            
            search_regions.append({
                'region': [new_x1, new_y1, new_x2, new_y2],
                'text': text,
                'confidence': region['confidence'],
                'original_location': location
            })
            
            print(f"   Created search region: [{new_x1}, {new_y1}, {new_x2}, {new_y2}] (Based on text: {text})")
        
        return search_regions
    
    def detect_scale_in_regions(self, image_path, search_regions):
        """
        Detect scale bars in specified regions
        
        Args:
            image_path: Path to image file
            search_regions: List of search regions
            
        Returns:
            list: List of detection results
        """
        results = []
        
        for region_info in search_regions:
            region = region_info['region']
            x1, y1, x2, y2 = region
            
            # Check if region is valid
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                print(f"    Skipping invalid search region: [{x1}, {y1}, {x2}, {y2}]")
                continue
            
            # Crop image region
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Ensure coordinates stay within image boundaries
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(x1+1, min(x2, width))
            y2 = max(y1+1, min(y2, height))
            
            cropped_image = image[y1:y2, x1:x2]
            
            # Check if cropped image is valid
            if cropped_image.size == 0:
                print(f"    Invalid cropped image: [{x1}, {y1}, {x2}, {y2}]")
                continue
            
            # Save temporary cropped image
            temp_path = f"temp_crop_{x1}_{y1}.jpg"
            cv2.imwrite(temp_path, cropped_image)
            
            try:
                # Detect in cropped region
                crop_results = self.model.predict(temp_path, conf=self.min_confidence, verbose=False)
                
                if len(crop_results) > 0 and len(crop_results[0].boxes) > 0:
                    # Get detection results and convert back to original image coordinates
                    for box in crop_results[0].boxes.xyxy:
                        # Convert coordinates
                        crop_x1, crop_y1, crop_x2, crop_y2 = box.cpu().numpy()
                        orig_x1 = x1 + crop_x1
                        orig_y1 = y1 + crop_y1
                        orig_x2 = x1 + crop_x2
                        orig_y2 = y1 + crop_y2
                        
                        results.append({
                            'bbox': [orig_x1, orig_y1, orig_x2, orig_y2],
                            'text': region_info['text'],
                            'confidence': region_info['confidence'],
                            'search_region': region,
                            'detection_method': 'ocr_guided'
                        })
                
            finally:
                # Delete temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return results
    
    def detect_scale_full_image(self, image_path, conf=0.2):
        """
        Detect scale bars in full image (as fallback option)
        
        Args:
            image_path: Path to image file
            conf: Confidence threshold
            
        Returns:
            list: List of detection results
        """
        results = self.model.predict(image_path, conf=conf, verbose=False)
        
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes.xyxy:
                detections.append({
                    'bbox': box.cpu().numpy().tolist(),
                    'text': 'Unknown',
                    'confidence': 0.5,
                    'search_region': 'full_image',
                    'detection_method': 'full_image'
                })
        
        return detections
    
    def detect_scale_hybrid(self, image_path, conf=0.2):
        """
        Hybrid detection strategy: OCR-guided first, then full image detection
        
        Args:
            image_path: Path to image file
            conf: Confidence threshold
            
        Returns:
            dict: Detection results
        """
        print(f" Hybrid detection for image: {os.path.basename(image_path)}")
        
        # 1. Detect text with OCR
        text_regions = self.detect_scale_text(image_path)
        
        # 2. Create search regions
        search_regions = self.create_search_regions(image_path, text_regions)
        
        # 3. Detect in search regions
        ocr_guided_results = self.detect_scale_in_regions(image_path, search_regions)
        print(f"   OCR-guided detection found {len(ocr_guided_results)} scale bars")
        
        # 4. Full image detection (fallback)
        full_image_results = self.detect_scale_full_image(image_path, conf)
        print(f"   Full image detection found {len(full_image_results)} scale bars")
        
        # 5. Merge results (prioritize OCR-guided results)
        final_results = ocr_guided_results + full_image_results
        
        return {
            'image_path': image_path,
            'text_regions': text_regions,
            'search_regions': search_regions,
            'ocr_guided_results': ocr_guided_results,
            'full_image_results': full_image_results,
            'final_results': final_results,
            'total_detections': len(final_results)
        }
    
    def visualize_results(self, detection_result, output_path=None):
        """
        Visualize detection results
        
        Args:
            detection_result: Detection results dictionary
            output_path: Output image path
        """
        image = cv2.imread(detection_result['image_path'])
        if image is None:
            return
        
        vis_image = image.copy()
        
        # Draw text regions
        for region in detection_result['text_regions']:
            loc = region['location']
            x1, y1, x2, y2 = loc['left'], loc['top'], loc['left'] + loc['width'], loc['top'] + loc['height']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue: text regions
            cv2.putText(vis_image, region['text'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw search regions
        for region_info in detection_result['search_regions']:
            x1, y1, x2, y2 = region_info['region']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Yellow: search regions
        
        # Draw detection results
        for result in detection_result['final_results']:
            x1, y1, x2, y2 = map(int, result['bbox'])
            
            # Select color based on detection method
            if result['detection_method'] == 'ocr_guided':
                color = (0, 255, 0)  # Green: OCR-guided detection
            else:
                color = (0, 0, 255)  # Red: full image detection
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_image, (x1, y1), 3, color, -1)  # Endpoint 1
            cv2.circle(vis_image, (x2, y2), 3, color, -1)  # Endpoint 2
            
            # Add text information
            text = f"{result['text']} ({result['detection_method']})"
            cv2.putText(vis_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save results
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f" Visualization results saved to: {output_path}")
        
        return vis_image

def test_ocr_guided_detector():
    """Test OCR-guided detector"""
    print(" Testing OCR-guided Scale Bar Detector")
    print("="*60)
    
    # Configuration
    model_path = "weights/epoch80.pt"
    images_dir = "inputs/lizi/images"
    output_dir = "ocr_guided_detection"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = OCRGuidedScaleDetector(model_path)
    
    # Test images
    test_images = ['100.png', '238.png', '318.png', '159.png']
    
    for img_name in test_images:
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f" Image does not exist: {img_path}")
            continue
        
        print(f"\n Processing image: {img_name}")
        
        # Perform hybrid detection
        result = detector.detect_scale_hybrid(img_path, conf=0.2)
        
        # Visualize results
        output_path = os.path.join(output_dir, f"{img_name.replace('.png', '')}_ocr_guided.png")
        detector.visualize_results(result, output_path)
        
        print(f"   Total detections: {result['total_detections']}")
        print(f"   OCR-guided detections: {len(result['ocr_guided_results'])}")
        print(f"   Full image detections: {len(result['full_image_results'])}")

def test_missed_images():
    """Test images that were previously undetected"""
    print(" Testing Previously Undetected Images")
    print("="*60)
    
    # Configuration
    model_path = "weights/epoch80.pt"
    images_dir = "inputs/lizi/images"
    output_dir = "ocr_guided_missed_detection"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    detector = OCRGuidedScaleDetector(model_path)
    
    # Images that were previously undetected
    missed_images = ['87.png', '167.png', '248.png', '271.png', '298.png', '299.png', 
                    '338.png', '391.png', '393.png', '398.png', '405.png', '451.png']
    
    for img_name in missed_images:
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f" Image does not exist: {img_path}")
            continue
        
        print(f"\n Processing undetected image: {img_name}")
        
        # Perform hybrid detection with lower confidence threshold
        result = detector.detect_scale_hybrid(img_path, conf=0.1)
        
        # Visualize results
        output_path = os.path.join(output_dir, f"{img_name.replace('.png', '')}_ocr_guided.png")
        detector.visualize_results(result, output_path)
        
        print(f"   Total detections: {result['total_detections']}")
        print(f"   OCR-guided detections: {len(result['ocr_guided_results'])}")
        print(f"   Full image detections: {len(result['full_image_results'])}")

if __name__ == '__main__':
    # Test regular images
    test_ocr_guided_detector()
    
    # Test previously undetected images
    test_missed_images()
