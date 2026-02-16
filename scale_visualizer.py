#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scale Bar Detector Visualization Module
Provides comprehensive visualization functionalities
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import torch
from typing import List, Tuple, Optional, Dict
import seaborn as sns
from lightweight_scale_detector import LightweightScaleDetectorWithPreprocessing
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ScaleVisualizer:
    """Scale Bar Detector Visualization Class"""
    
    def __init__(self, detector: LightweightScaleDetectorWithPreprocessing):
        self.detector = detector
        
    def visualize_detection_with_details(self, 
                                       image: np.ndarray, 
                                       coords: List[int], 
                                       length: float, 
                                       confidence: float,
                                       save_path: str = 'detailed_detection.jpg') -> np.ndarray:
        """
        Visualize detection results with detailed information
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        vis_image = image.copy()
        if coords is not None:
            x1, y1, x2, y2 = coords
            
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            cv2.circle(vis_image, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(vis_image, (x2, y2), 8, (255, 0, 0), -1)
            
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            dx, dy = x2 - x1, y2 - y1
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 0:
                dx, dy = dx/norm * 20, dy/norm * 20
                cv2.arrowedLine(vis_image, (mid_x, mid_y), 
                               (int(mid_x + dx), int(mid_y + dy)), 
                               (0, 255, 255), 3, tipLength=0.3)
            
            info_text = [
                f"Pixel Length: {length:.1f}px",
                f"Confidence: {confidence:.3f}",
                f"Start Point: ({x1}, {y1})",
                f"End Point: ({x2}, {y2})"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(vis_image, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        axes[1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Detection Results', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return vis_image
    
    def visualize_multiple_detections(self, 
                                    images: List[np.ndarray], 
                                    results: List[Tuple],
                                    save_path: str = 'multiple_detections.jpg'):
        """
        Visualize multiple detection results in a grid layout
        """
        n_images = len(images)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (image, result) in enumerate(zip(images, results)):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            vis_image = image.copy()
            coords, length, confidence = result
            
            if coords is not None:
                x1, y1, x2, y2 = coords
                cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.circle(vis_image, (x1, y1), 5, (255, 0, 0), -1)
                cv2.circle(vis_image, (x2, y2), 5, (255, 0, 0), -1)
                
                cv2.putText(vis_image, f"{length:.0f}px", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            ax.set_title(f'Image {i+1}', fontsize=12)
            ax.axis('off')
        
        for i in range(n_images, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_confidence_distribution(self, 
                                        confidences: List[float],
                                        save_path: str = 'confidence_distribution.jpg'):
        """
        Visualize confidence score distribution with histogram and box plot
        """
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Score Distribution Histogram')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(confidences, patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
        plt.ylabel('Confidence Score')
        plt.title('Confidence Score Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Confidence Score Statistics:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        print(f"  Standard Deviation: {np.std(confidences):.3f}")
        print(f"  Minimum: {np.min(confidences):.3f}")
        print(f"  Maximum: {np.max(confidences):.3f}")
    
    def visualize_length_distribution(self, 
                                    lengths: List[float],
                                    save_path: str = 'length_distribution.jpg'):
        """
        Visualize scale bar length distribution with multiple plots
        """
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.hist(lengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Pixel Length')
        plt.ylabel('Frequency')
        plt.title('Length Distribution Histogram')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.scatter(range(len(lengths)), lengths, alpha=0.6, color='red')
        plt.xlabel('Sample Index')
        plt.ylabel('Pixel Length')
        plt.title('Length Scatter Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.boxplot(lengths, patch_artist=True, 
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
        plt.ylabel('Pixel Length')
        plt.title('Length Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Length Statistics:")
        print(f"  Mean: {np.mean(lengths):.1f}px")
        print(f"  Median: {np.median(lengths):.1f}px")
        print(f"  Standard Deviation: {np.std(lengths):.1f}px")
        print(f"  Minimum: {np.min(lengths):.1f}px")
        print(f"  Maximum: {np.max(lengths):.1f}px")
    
    def create_interactive_demo(self, image_path: str):
        """
        Create interactive demo with threshold analysis
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            return
        
        coords, length, confidence = self.detector.detect_scale(image)
        
        self.visualize_detection_with_details(image, coords, length, confidence)
        
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        for threshold in thresholds:
            coords, length, conf = self.detector.detect_scale(image, confidence_threshold=threshold)
            results.append((threshold, conf if conf is not None else 0))
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        thresholds_list, confidences_list = zip(*results)
        plt.plot(thresholds_list, confidences_list, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Detection Confidence')
        plt.title('Threshold Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(thresholds_list, confidences_list, alpha=0.7, color='lightblue')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Detection Confidence')
        plt.title('Threshold Analysis (Bar Chart)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_analysis.jpg', dpi=300, bbox_inches='tight')
        plt.show()


def visualize_training_process(losses: List[float], save_path: str = 'training_process.jpg'):
    """
    Visualize model training process with loss curves
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(losses) > 1:
        loss_changes = np.diff(losses)
        plt.plot(loss_changes, 'r-', linewidth=2, label='Loss Change')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Change')
        plt.title('Loss Change Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_comparison_visualization(original_image: np.ndarray, 
                                  detected_coords: List[int],
                                  ground_truth_coords: Optional[List[int]] = None,
                                  save_path: str = 'comparison.jpg'):
    """
    Create comparison visualization (detection results vs ground truth)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    vis_image = original_image.copy()
    
    if detected_coords is not None:
        x1, y1, x2, y2 = detected_coords
        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.circle(vis_image, (x1, y1), 8, (0, 255, 0), -1)
        cv2.circle(vis_image, (x2, y2), 8, (0, 255, 0), -1)
    
    if ground_truth_coords is not None:
        x1, y1, x2, y2 = ground_truth_coords
        cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.circle(vis_image, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(vis_image, (x2, y2), 8, (255, 0, 0), -1)
    
    axes[1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Detection Results Comparison', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    if detected_coords is not None and ground_truth_coords is not None:
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=4, label='Detected Result'),
            plt.Line2D([0], [0], color='red', lw=4, label='Ground Truth')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    detector = LightweightScaleDetectorWithPreprocessing(
        model_path='lightweight_scale_detector.pth',
        input_size=(320, 320),
        hidden_dim=32
    )
    
    visualizer = ScaleVisualizer(detector)
    
    test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    cv2.line(test_image, (50, 50), (250, 150), (255, 255, 255), 8)
    
    coords, length, confidence = detector.detect_scale(test_image)
    
    visualizer.visualize_detection_with_details(test_image, coords, length, confidence)
    
    print("Visualization completed!")
