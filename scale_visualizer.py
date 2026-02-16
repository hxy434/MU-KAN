#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比例尺检测器可视化模块
提供丰富的可视化功能
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

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ScaleVisualizer:
    """比例尺检测器可视化类"""
    
    def __init__(self, detector: LightweightScaleDetectorWithPreprocessing):
        self.detector = detector
        
    def visualize_detection_with_details(self, 
                                       image: np.ndarray, 
                                       coords: List[int], 
                                       length: float, 
                                       confidence: float,
                                       save_path: str = 'detailed_detection.jpg') -> np.ndarray:
        """
        详细的可视化检测结果
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 原始图像
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('原始图像', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 检测结果
        vis_image = image.copy()
        if coords is not None:
            x1, y1, x2, y2 = coords
            
            # 绘制比例尺线段
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # 绘制端点
            cv2.circle(vis_image, (x1, y1), 8, (255, 0, 0), -1)
            cv2.circle(vis_image, (x2, y2), 8, (255, 0, 0), -1)
            
            # 添加箭头指示方向
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            dx, dy = x2 - x1, y2 - y1
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 0:
                dx, dy = dx/norm * 20, dy/norm * 20
                cv2.arrowedLine(vis_image, (mid_x, mid_y), 
                               (int(mid_x + dx), int(mid_y + dy)), 
                               (0, 255, 255), 3, tipLength=0.3)
            
            # 添加文本信息
            info_text = [
                f"像素长度: {length:.1f}px",
                f"置信度: {confidence:.3f}",
                f"起点: ({x1}, {y1})",
                f"终点: ({x2}, {y2})"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(vis_image, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        axes[1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('检测结果', fontsize=14, fontweight='bold')
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
        可视化多个检测结果
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
                
                # 添加长度信息
                cv2.putText(vis_image, f"{length:.0f}px", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            ax.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            ax.set_title(f'图像 {i+1}', fontsize=12)
            ax.axis('off')
        
        # 隐藏多余的子图
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
        可视化置信度分布
        """
        plt.figure(figsize=(10, 6))
        
        # 直方图
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('置信度')
        plt.ylabel('频次')
        plt.title('置信度分布直方图')
        plt.grid(True, alpha=0.3)
        
        # 箱线图
        plt.subplot(1, 2, 2)
        plt.boxplot(confidences, patch_artist=True, 
                   boxprops=dict(facecolor='lightgreen', alpha=0.7))
        plt.ylabel('置信度')
        plt.title('置信度箱线图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"置信度统计:")
        print(f"  平均值: {np.mean(confidences):.3f}")
        print(f"  中位数: {np.median(confidences):.3f}")
        print(f"  标准差: {np.std(confidences):.3f}")
        print(f"  最小值: {np.min(confidences):.3f}")
        print(f"  最大值: {np.max(confidences):.3f}")
    
    def visualize_length_distribution(self, 
                                    lengths: List[float],
                                    save_path: str = 'length_distribution.jpg'):
        """
        可视化长度分布
        """
        plt.figure(figsize=(12, 5))
        
        # 直方图
        plt.subplot(1, 3, 1)
        plt.hist(lengths, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('像素长度')
        plt.ylabel('频次')
        plt.title('长度分布直方图')
        plt.grid(True, alpha=0.3)
        
        # 散点图（长度 vs 索引）
        plt.subplot(1, 3, 2)
        plt.scatter(range(len(lengths)), lengths, alpha=0.6, color='red')
        plt.xlabel('样本索引')
        plt.ylabel('像素长度')
        plt.title('长度散点图')
        plt.grid(True, alpha=0.3)
        
        # 箱线图
        plt.subplot(1, 3, 3)
        plt.boxplot(lengths, patch_artist=True, 
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
        plt.ylabel('像素长度')
        plt.title('长度箱线图')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"长度统计:")
        print(f"  平均值: {np.mean(lengths):.1f}px")
        print(f"  中位数: {np.median(lengths):.1f}px")
        print(f"  标准差: {np.std(lengths):.1f}px")
        print(f"  最小值: {np.min(lengths):.1f}px")
        print(f"  最大值: {np.max(lengths):.1f}px")
    
    def create_interactive_demo(self, image_path: str):
        """
        创建交互式演示
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 检测比例尺
        coords, length, confidence = self.detector.detect_scale(image)
        
        # 创建详细可视化
        self.visualize_detection_with_details(image, coords, length, confidence)
        
        # 创建置信度阈值分析
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = []
        
        for threshold in thresholds:
            coords, length, conf = self.detector.detect_scale(image, confidence_threshold=threshold)
            results.append((threshold, conf if conf is not None else 0))
        
        # 绘制阈值分析
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        thresholds_list, confidences_list = zip(*results)
        plt.plot(thresholds_list, confidences_list, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('置信度阈值')
        plt.ylabel('检测置信度')
        plt.title('阈值分析')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(thresholds_list, confidences_list, alpha=0.7, color='lightblue')
        plt.xlabel('置信度阈值')
        plt.ylabel('检测置信度')
        plt.title('阈值分析（柱状图）')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_analysis.jpg', dpi=300, bbox_inches='tight')
        plt.show()


def visualize_training_process(losses: List[float], save_path: str = 'training_process.jpg'):
    """
    可视化训练过程
    """
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses, 'b-', linewidth=2, label='训练损失')
    plt.xlabel('训练步数')
    plt.ylabel('损失值')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 损失变化率
    plt.subplot(1, 2, 2)
    if len(losses) > 1:
        loss_changes = np.diff(losses)
        plt.plot(loss_changes, 'r-', linewidth=2, label='损失变化')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('训练步数')
        plt.ylabel('损失变化')
        plt.title('损失变化率')
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
    创建对比可视化（检测结果 vs 真实标签）
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始图像
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始图像', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 检测结果对比
    vis_image = original_image.copy()
    
    # 绘制检测结果（绿色）
    if detected_coords is not None:
        x1, y1, x2, y2 = detected_coords
        cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.circle(vis_image, (x1, y1), 8, (0, 255, 0), -1)
        cv2.circle(vis_image, (x2, y2), 8, (0, 255, 0), -1)
    
    # 绘制真实标签（红色）
    if ground_truth_coords is not None:
        x1, y1, x2, y2 = ground_truth_coords
        cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.circle(vis_image, (x1, y1), 8, (255, 0, 0), -1)
        cv2.circle(vis_image, (x2, y2), 8, (255, 0, 0), -1)
    
    axes[1].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('检测结果对比', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # 添加图例
    if detected_coords is not None and ground_truth_coords is not None:
        legend_elements = [
            plt.Line2D([0], [0], color='green', lw=4, label='检测结果'),
            plt.Line2D([0], [0], color='red', lw=4, label='真实标签')
        ]
        axes[1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 加载检测器
    detector = LightweightScaleDetectorWithPreprocessing(
        model_path='lightweight_scale_detector.pth',
        input_size=(320, 320),
        hidden_dim=32
    )
    
    # 创建可视化器
    visualizer = ScaleVisualizer(detector)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    cv2.line(test_image, (50, 50), (250, 150), (255, 255, 255), 8)
    
    # 检测比例尺
    coords, length, confidence = detector.detect_scale(test_image)
    
    # 详细可视化
    visualizer.visualize_detection_with_details(test_image, coords, length, confidence)
    
    print("🎨 可视化完成！") 