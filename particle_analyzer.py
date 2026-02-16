#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子分析模块
在比例尺检测基础上，检测图像中的粒子并生成像素分布图
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
from typing import List, Tuple, Dict, Optional
from lightweight_scale_detector import LightweightScaleDetectorWithPreprocessing
from scale_visualizer import ScaleVisualizer
import seaborn as sns
from scipy import ndimage
from skimage import measure, morphology
from skimage.filters import threshold_otsu, gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd
from mock_ocr import process as ocr_process
import os

class ParticleAnalyzer:
    """粒子分析器"""
    
    def __init__(self, scale_detector: LightweightScaleDetectorWithPreprocessing):
        self.scale_detector = scale_detector
        self.scale_length_pixels = None  # 比例尺像素长度
        self.scale_length_real = None    # 比例尺实际长度（单位：微米等）
        self.pixel_to_real_ratio = None  # 像素到实际单位的转换比例
        
    def set_scale_reference(self, scale_length_real: float, unit: str = "μm"):
        """
        设置比例尺参考
        Args:
            scale_length_real: 比例尺的实际长度
            unit: 单位（微米、毫米等）
        """
        self.scale_length_real = scale_length_real
        self.unit = unit
        
    def detect_particles(self, 
                        image: np.ndarray, 
                        method: str = "watershed",
                        min_area: int = 50,
                        max_area: int = 5000,
                        circularity_threshold: float = 0.3,
                        ocr_boxes: list = None) -> Dict:
        """
        检测图像中的粒子
        Args:
            image: 输入图像
            method: 检测方法 ("watershed", "contour", "blob")
            min_area: 最小粒子面积
            max_area: 最大粒子面积
            circularity_threshold: 圆度阈值
        Returns:
            包含粒子信息的字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 预处理
        blurred = gaussian(gray, sigma=1)
        
        if method == "watershed":
            particles = self._detect_particles_watershed(blurred, min_area, max_area, ocr_boxes)
        elif method == "contour":
            particles = self._detect_particles_contour(blurred, min_area, max_area, circularity_threshold)
        elif method == "blob":
            particles = self._detect_particles_blob(blurred, min_area, max_area)
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        return particles
    
    def _detect_particles_watershed(self, 
                                  image: np.ndarray, 
                                  min_area: int, 
                                  max_area: int,
                                  ocr_boxes: list = None) -> Dict:
        """使用分水岭算法检测粒子"""
        # 二值化 - 尝试多种方法
        try:
            thresh = threshold_otsu(image)
            binary = image < thresh
        except:
            # 如果Otsu失败，使用固定阈值
            binary = image < 128
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = morphology.binary_opening(binary, kernel)
        binary = morphology.binary_closing(binary, kernel)
        
        # 距离变换
        distance = ndimage.distance_transform_edt(binary)
        
        # 找到局部最大值
        local_max = peak_local_max(distance, min_distance=10, labels=binary)
        local_max_mask = np.zeros_like(binary, dtype=bool)
        local_max_mask[tuple(local_max.T)] = True
        
        # 标记
        markers = measure.label(local_max_mask)
        
        # 分水岭分割
        labels = watershed(-distance, markers, mask=binary)
        
        # 分析区域
        regions = measure.regionprops(labels)
        
        particles = {
            'centroids': [],
            'areas': [],
            'diameters': [],
            'circularities': [],
            'bboxes': [],
            'labels': []
        }
        
        # 过滤粒子时，排除与ocr_boxes重叠的轮廓
        filtered = 0
        for region in regions:
            if min_area <= region.area <= max_area:
                # 计算外接矩形
                minr, minc, maxr, maxc = region.bbox
                region_box = (minc, minr, maxc, maxr)
                overlap = False
                if ocr_boxes:
                    for ocr_box in ocr_boxes:
                        # 调试输出
                        print(f"[调试] region_box: {region_box}, ocr_box: {ocr_box}")
                        # 判断是否有重叠
                        if not (region_box[2] < ocr_box[0] or ocr_box[2] < region_box[0] or region_box[3] < ocr_box[1] or ocr_box[3] < region_box[1]):
                            overlap = True
                            print(f"[调试] overlap=True, 该轮廓被排除")
                            break
                if not overlap:
                    # 只保留未与OCR box重叠的粒子
                    particles['centroids'].append(region.centroid)
                    particles['areas'].append(region.area)
                    particles['diameters'].append(region.equivalent_diameter)
                    particles['circularities'].append(region.extent)
                    particles['bboxes'].append(region.bbox)
                    particles['labels'].append(region.label)
                else:
                    filtered += 1
        if ocr_boxes:
            print(f"[过滤] 有 {filtered} 个轮廓因与OCR box重叠被排除")
        
        return particles
    
    def _detect_particles_contour(self, 
                                image: np.ndarray, 
                                min_area: int, 
                                max_area: int,
                                circularity_threshold: float) -> Dict:
        """使用轮廓检测粒子"""
        # 二值化 - 尝试多种方法
        try:
            thresh = threshold_otsu(image)
            binary = image < thresh
        except:
            # 如果Otsu失败，使用固定阈值
            binary = image < 128
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        binary = morphology.binary_opening(binary, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        particles = {
            'centroids': [],
            'areas': [],
            'diameters': [],
            'circularities': [],
            'bboxes': [],
            'contours': []
        }
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # 计算圆度
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                if circularity >= circularity_threshold:
                    # 计算质心
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    
                    # 计算等效直径
                    diameter = np.sqrt(4 * area / np.pi)
                    
                    # 边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    particles['centroids'].append((cy, cx))  # OpenCV使用(y, x)
                    particles['areas'].append(area)
                    particles['diameters'].append(diameter)
                    particles['circularities'].append(circularity)
                    particles['bboxes'].append((y, x, y+h, x+w))
                    particles['contours'].append(contour)
        
        return particles
    
    def _detect_particles_blob(self, 
                             image: np.ndarray, 
                             min_area: int, 
                             max_area: int) -> Dict:
        """使用斑点检测"""
        # 确保图像是8位
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 使用SimpleBlobDetector
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByArea = True
        params.filterByCircularity = True
        params.minCircularity = 0.3
        params.filterByConvexity = True
        params.minConvexity = 0.5
        
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image)
        
        particles = {
            'centroids': [],
            'areas': [],
            'diameters': [],
            'circularities': [],
            'bboxes': [],
            'keypoints': keypoints
        }
        
        for kp in keypoints:
            x, y = kp.pt
            size = kp.size
            
            particles['centroids'].append((y, x))
            particles['areas'].append(np.pi * (size/2)**2)
            particles['diameters'].append(size)
            particles['circularities'].append(1.0)  # 假设为圆形
            particles['bboxes'].append((y-size/2, x-size/2, y+size/2, x+size/2))
        
        return particles
    
    def analyze_particle_distribution(self, particles: Dict) -> Dict:
        """分析粒子分布"""
        if not particles['areas']:
            return {}
        
        areas = np.array(particles['areas'])
        diameters = np.array(particles['diameters'])
        circularities = np.array(particles['circularities'])
        
        # 转换为实际单位
        if self.pixel_to_real_ratio:
            areas_real = areas * (self.pixel_to_real_ratio ** 2)
            diameters_real = diameters * self.pixel_to_real_ratio
        else:
            areas_real = areas
            diameters_real = diameters
        
        analysis = {
            'count': len(areas),
            'area_stats': {
                'mean': np.mean(areas_real),
                'std': np.std(areas_real),
                'min': np.min(areas_real),
                'max': np.max(areas_real),
                'median': np.median(areas_real)
            },
            'diameter_stats': {
                'mean': np.mean(diameters_real),
                'std': np.std(diameters_real),
                'min': np.min(diameters_real),
                'max': np.max(diameters_real),
                'median': np.median(diameters_real)
            },
            'circularity_stats': {
                'mean': np.mean(circularities),
                'std': np.std(circularities),
                'min': np.min(circularities),
                'max': np.max(circularities)
            },
            'size_distribution': {
                'small': np.sum(diameters_real < np.percentile(diameters_real, 33)),
                'medium': np.sum((diameters_real >= np.percentile(diameters_real, 33)) & 
                               (diameters_real < np.percentile(diameters_real, 67))),
                'large': np.sum(diameters_real >= np.percentile(diameters_real, 67))
            }
        }
        
        return analysis
    
    def visualize_particle_distribution(self, 
                                      image: np.ndarray, 
                                      particles: Dict, 
                                      analysis: Dict,
                                      save_path: str = 'particle_distribution.jpg',
                                      ocr_boxes: list = None):
        """可视化粒子分布"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Particle Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 原始图像和检测结果
        ax1 = plt.subplot(2, 4, 1)
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image', fontsize=12)
        ax1.axis('off')
        
        # 检测结果
        ax2 = plt.subplot(2, 4, 2)
        vis_image = image.copy()
        for i, (centroid, bbox) in enumerate(zip(particles['centroids'], particles['bboxes'])):
            y, x = centroid
            y1, x1, y2, x2 = bbox
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(vis_image, (int(x), int(y)), 3, (255, 0, 0), -1)
        ax2.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Detection Results ({len(particles["areas"])} particles)', fontsize=12)
        ax2.axis('off')
        
        # 在原图和检测结果上画出ocr_boxes
        if ocr_boxes:
            for ocr_box in ocr_boxes:
                ax2.add_patch(
                    plt.Rectangle((ocr_box[0], ocr_box[1]), ocr_box[2]-ocr_box[0], ocr_box[3]-ocr_box[1],
                                  fill=False, edgecolor='purple', linewidth=2, linestyle='--', alpha=0.8))
            print(f"[可视化] 已画出 {len(ocr_boxes)} 个OCR box（紫色框）")
        # 画出所有粒子的bbox（蓝色）
        if 'bboxes' in particles:
            for bbox in particles['bboxes']:
                minr, minc, maxr, maxc = bbox
                ax2.add_patch(
                    plt.Rectangle((minc, minr), maxc-minc, maxr-minr,
                                  fill=False, edgecolor='blue', linewidth=1, linestyle=':', alpha=0.7))
            print(f"[可视化] 已画出 {len(particles['bboxes'])} 个粒子bbox（蓝色框）")
        
        # 面积分布直方图
        ax3 = plt.subplot(2, 4, 3)
        if particles['areas']:
            areas = np.array(particles['areas'])
            if self.pixel_to_real_ratio:
                areas = areas * (self.pixel_to_real_ratio ** 2)
            plt.hist(areas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Area ($\\mu m^2$)' if self.pixel_to_real_ratio else 'Area (pixels²)')
            plt.ylabel('Frequency')
            plt.title('Area Distribution')
        
        # 直径分布直方图
        ax4 = plt.subplot(2, 4, 4)
        if particles['diameters']:
            diameters = np.array(particles['diameters'])
            if self.pixel_to_real_ratio:
                diameters = diameters * self.pixel_to_real_ratio
            plt.hist(diameters, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.xlabel(f'Diameter ({self.unit})' if self.pixel_to_real_ratio else 'Diameter (pixels)')
            plt.ylabel('Frequency')
            plt.title('Diameter Distribution')
        
        # 圆度分布
        ax5 = plt.subplot(2, 4, 5)
        if particles['circularities']:
            plt.hist(particles['circularities'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.xlabel('Circularity')
            plt.ylabel('Frequency')
            plt.title('Circularity Distribution')
        
        # 尺寸分类饼图
        ax6 = plt.subplot(2, 4, 6)
        if analysis and 'size_distribution' in analysis:
            sizes = analysis['size_distribution']
            labels = ['小粒子', '中粒子', '大粒子']
            values = [sizes['small'], sizes['medium'], sizes['large']]
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.title('尺寸分类')
        
        # 统计信息
        ax7 = plt.subplot(2, 4, 7)
        if analysis:
            stats_text = [
                f"总粒子数: {analysis['count']}",
                f"平均直径: {analysis['diameter_stats']['mean']:.2f}",
                f"直径标准差: {analysis['diameter_stats']['std']:.2f}",
                f"平均面积: {analysis['area_stats']['mean']:.2f}",
                f"面积标准差: {analysis['area_stats']['std']:.2f}",
                f"平均圆度: {analysis['circularity_stats']['mean']:.3f}"
            ]
            if self.pixel_to_real_ratio:
                stats_text = [s + f" {self.unit}" if "直径" in s or "面积" in s else s for s in stats_text]
            
            plt.text(0.1, 0.9, '\n'.join(stats_text), transform=ax7.transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax7.axis('off')
            ax7.set_title('统计信息')
        
        # 散点图：面积 vs 圆度
        ax8 = plt.subplot(2, 4, 8)
        if particles['areas'] and particles['circularities']:
            areas = np.array(particles['areas'])
            if self.pixel_to_real_ratio:
                areas = areas * (self.pixel_to_real_ratio ** 2)
            plt.scatter(areas, particles['circularities'], alpha=0.6, color='purple')
            plt.xlabel('Area ($\\mu m^2$)' if self.pixel_to_real_ratio else 'Area (pixels²)')
            plt.ylabel('Circularity')
            plt.title('Area vs Circularity')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return vis_image
    
    def analyze_image_with_scale(self, 
                                image: np.ndarray,
                                scale_length_real: float = None,
                                unit: str = "μm",
                                particle_method: str = "watershed",
                                save_path: str = "particle_analysis.jpg",
                                image_path: str = None,
                                access_token: str = None,
                                ocr_boxes: list = None) -> Dict:
        """
        完整的图像分析流程：检测比例尺 + OCR识别 + 分析粒子
        """
        print("🔍 开始图像分析...")
        
        # 1. 如果有ocr_boxes，先mask掉这些区域
        if ocr_boxes and len(ocr_boxes) > 0:
            print(f"[mask] 将抹白 {len(ocr_boxes)} 个OCR box区域")
            image = image.copy()
            for box in ocr_boxes:
                x1, y1, x2, y2 = box
                image[y1:y2, x1:x2, :] = 255  # 抹白
        
        # 1. 检测比例尺（如果有检测器）
        coords = None
        scale_length_pixels = None
        confidence = None
        pixel_length = None
        
        if self.scale_detector is not None:
            print("1. 检测比例尺...")
            coords, scale_length_pixels, confidence = self.scale_detector.detect_scale(image)
            
            if coords is not None:
                self.scale_length_pixels = scale_length_pixels
                print(f"2. 比例尺检测: {scale_length_pixels:.1f}像素")
                
                # 2. OCR自动识别物理长度（如果提供了图像路径和access_token）
                if image_path and access_token and scale_length_real is None:
                    print("3. OCR识别比例尺文本...")
                    try:
                        ocr_result = ocr_process(image_path, access_token)
                        print("   OCR原始结果：", ocr_result)
                        
                        words_result = ocr_result.get("words_result", [])
                        for item in words_result:
                            words = item["words"].replace(' ', '')
                            for unit_candidate in ["nm", "μm", "um"]:
                                if unit_candidate in words:
                                    num_str = words.replace(unit_candidate, "")
                                    try:
                                        value = float(num_str)
                                        scale_length_real = value
                                        unit = unit_candidate
                                        print(f"   OCR识别结果: {scale_length_real} {unit}")
                                        break
                                    except:
                                        continue
                            if scale_length_real:
                                break
                    except Exception as e:
                        print(f"   OCR识别失败: {e}")
                
                # 3. 设置比例尺参考
                if scale_length_real:
                    self.set_scale_reference(scale_length_real, unit)
                    self.pixel_to_real_ratio = scale_length_real / scale_length_pixels
                    pixel_length = self.pixel_to_real_ratio
                    # 新增详细调试输出
                    print(f"[调试] 比例尺像素长度: {scale_length_pixels}")
                    print(f"[调试] OCR识别物理长度: {scale_length_real} {unit}")
                    print(f"[调试] 每像素物理长度: {pixel_length} {unit}")
                    print(f"4. 比例尺设置: {scale_length_pixels:.1f}像素 = {scale_length_real}{unit}")
                    print(f"   每像素长度: {pixel_length:.4f}{unit}")
                else:
                    print(f"4. 比例尺检测: {scale_length_pixels:.1f}像素")
            else:
                print("⚠️  未检测到比例尺，将使用像素单位进行分析")
        else:
            print("⚠️  未提供比例尺检测器，将使用像素单位进行分析")
        
        if pixel_length is None:
            print("   每像素长度: 未知（仅以像素为单位）")
        
        # 5. 检测粒子
        print("5. 检测粒子...")
        particles = self.detect_particles(image, method=particle_method, ocr_boxes=ocr_boxes)
        print(f"   检测到 {len(particles['areas'])} 个粒子")
        
        # 6. 分析粒子分布
        print("6. 分析粒子分布...")
        analysis = self.analyze_particle_distribution(particles)
        
        # 7. 可视化结果
        print("7. 生成可视化...")
        self.visualize_particle_distribution(image, particles, analysis, save_path, ocr_boxes)
        
        # 8. 保存详细数据
        self.save_analysis_data(particles, analysis, save_path.replace('.jpg', '_data.csv'), pixel_length, unit)
        
        print("✅ 分析完成！")
        
        return {
            'scale_info': {
                'coords': coords,
                'length_pixels': scale_length_pixels,
                'length_real': scale_length_real,
                'confidence': confidence,
                'unit': unit,
                'pixel_length': pixel_length
            },
            'particles': particles,
            'analysis': analysis,
            'pixel_length': pixel_length
        }
    
    def save_analysis_data(self, particles: Dict, analysis: Dict, csv_path: str, pixel_length: float = None, unit: str = "μm"):
        """保存分析数据到CSV文件，并在首行增加每像素长度说明"""
        if not particles['areas']:
            return
        
        # 创建DataFrame
        data = {
            'Particle_ID': range(1, len(particles['areas']) + 1),
            'Centroid_X': [c[1] for c in particles['centroids']],
            'Centroid_Y': [c[0] for c in particles['centroids']],
            'Area_Pixels': particles['areas'],
            'Diameter_Pixels': particles['diameters'],
            'Circularity': particles['circularities']
        }
        
        if self.pixel_to_real_ratio:
            data['Area_Real'] = [a * (self.pixel_to_real_ratio ** 2) for a in particles['areas']]
            data['Diameter_Real'] = [d * self.pixel_to_real_ratio for d in particles['diameters']]
        
        import pandas as pd
        df = pd.DataFrame(data)
        # 写入每像素长度说明
        with open(csv_path, 'w', encoding='utf-8') as f:
            if pixel_length:
                f.write(f'# 每像素长度: {pixel_length:.6f}{unit}\n')
            else:
                f.write(f'# 每像素长度: 未知（仅以像素为单位）\n')
            df.to_csv(f, index=False)
        print(f"📊 数据已保存到: {csv_path}")


# 使用示例
if __name__ == "__main__":
    # 加载比例尺检测器
    scale_detector = LightweightScaleDetectorWithPreprocessing(
        model_path='lightweight_scale_detector.pth',
        input_size=(320, 320),
        hidden_dim=32
    )
    
    # 创建粒子分析器
    analyzer = ParticleAnalyzer(scale_detector)
    
    # 创建测试图像（包含粒子和比例尺）
    test_image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    # 添加比例尺
    cv2.line(test_image, (50, 50), (250, 50), (255, 255, 255), 5)
    
    # 添加一些模拟粒子
    for i in range(20):
        x = np.random.randint(60, 280)
        y = np.random.randint(60, 280)
        radius = np.random.randint(5, 15)
        cv2.circle(test_image, (x, y), radius, (200, 200, 200), -1)
    
    # 分析图像
    results = analyzer.analyze_image_with_scale(
        test_image,
        scale_length_real=100,  # 假设比例尺代表100微米
        unit="μm",
        particle_method="watershed"
    )
    
    print("🎉 粒子分析完成！") 