#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于OCR引导的比例尺检测器
先识别比例尺的数字和单位，然后以该区域为中心扩展搜索比例尺
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
        初始化OCR引导的比例尺检测器
        
        Args:
            model_path: YOLO模型路径
            access_token: 百度OCR access_token
        """
        self.model = YOLO(model_path)
        self.access_token = access_token
        
        # 比例尺单位模式
        self.scale_units = ['um', 'μm', 'nm', 'mm', 'cm', 'm', 'pixel', 'px']
        self.scale_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(' + '|'.join(self.scale_units) + ')', re.IGNORECASE)
        
        # 扩展搜索参数
        self.expansion_ratio = 3.0  # 扩展倍数
        self.min_confidence = 0.1   # 扩展搜索时的最低置信度
        
        print("✅ OCR引导的比例尺检测器初始化完成")
        
    def detect_scale_text(self, image_path):
        """
        检测图片中的比例尺文本
        
        Args:
            image_path: 图片路径
            
        Returns:
            list: 检测到的文本区域列表，每个元素包含文本、位置、置信度
        """
        print(f"🔍 OCR检测文本: {os.path.basename(image_path)}")
        
        # 使用mock_ocr进行文本检测
        ocr_result = mock_ocr.process(image_path, self.access_token)
        
        if ocr_result.get("error_code", 0) != 0:
            print(f"❌ OCR检测失败: {ocr_result.get('error_msg', 'Unknown error')}")
            return []
        
        text_regions = []
        words_result = ocr_result.get("words_result", [])
        
        for item in words_result:
            text = item.get("words", "")
            location = item.get("location", {})
            
            # 检查是否包含比例尺模式
            if self.scale_pattern.search(text):
                # 确保位置信息完整
                if all(key in location for key in ['left', 'top', 'width', 'height']):
                    text_regions.append({
                        'text': text,
                        'location': location,
                        'confidence': 0.9  # OCR置信度
                    })
                    print(f"   ✅ 检测到比例尺文本: {text} 位置: {location}")
                else:
                    print(f"   ⚠️ 比例尺文本位置信息不完整: {text} 位置: {location}")
        
        print(f"   共检测到 {len(text_regions)} 个比例尺文本")
        return text_regions
    
    def create_search_regions(self, image_path, text_regions):
        """
        基于检测到的文本创建搜索区域
        
        Args:
            image_path: 图片路径
            text_regions: 检测到的文本区域列表
            
        Returns:
            list: 扩展后的搜索区域列表
        """
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        height, width = image.shape[:2]
        search_regions = []
        
        for region in text_regions:
            location = region['location']
            text = region['text']
            
            # 获取文本区域
            x1 = location['left']
            y1 = location['top']
            x2 = x1 + location['width']
            y2 = y1 + location['height']
            
            # 计算扩展区域
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 扩展搜索区域
            expanded_width = int(location['width'] * self.expansion_ratio)
            expanded_height = int(location['height'] * self.expansion_ratio)
            
            # 确保不超出图片边界
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
            
            print(f"   创建搜索区域: [{new_x1}, {new_y1}, {new_x2}, {new_y2}] (基于文本: {text})")
        
        return search_regions
    
    def detect_scale_in_regions(self, image_path, search_regions):
        """
        在指定区域内检测比例尺
        
        Args:
            image_path: 图片路径
            search_regions: 搜索区域列表
            
        Returns:
            list: 检测结果列表
        """
        results = []
        
        for region_info in search_regions:
            region = region_info['region']
            x1, y1, x2, y2 = region
            
            # 检查区域是否有效
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                print(f"   ⚠️ 跳过无效搜索区域: [{x1}, {y1}, {x2}, {y2}]")
                continue
            
            # 裁剪图片区域
            image = cv2.imread(image_path)
        if image is None:
                continue
            
            # 确保不超出图片边界
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(x1+1, min(x2, width))
            y2 = max(y1+1, min(y2, height))
            
            cropped_image = image[y1:y2, x1:x2]
            
            # 检查裁剪图片是否有效
            if cropped_image.size == 0:
                print(f"   ⚠️ 裁剪图片无效: [{x1}, {y1}, {x2}, {y2}]")
                continue
            
            # 保存临时裁剪图片
            temp_path = f"temp_crop_{x1}_{y1}.jpg"
            cv2.imwrite(temp_path, cropped_image)
            
            try:
                # 在裁剪区域中检测
                crop_results = self.model.predict(temp_path, conf=self.min_confidence, verbose=False)
                
                if len(crop_results) > 0 and len(crop_results[0].boxes) > 0:
                    # 获取检测结果并转换回原图坐标
                    for box in crop_results[0].boxes.xyxy:
                        # 转换坐标
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
                # 删除临时文件
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return results
    
    def detect_scale_full_image(self, image_path, conf=0.2):
        """
        在全图中检测比例尺（作为备选方案）
        
        Args:
            image_path: 图片路径
            conf: 置信度阈值
            
        Returns:
            list: 检测结果列表
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
        混合检测策略：先OCR引导，再全图检测
        
        Args:
            image_path: 图片路径
            conf: 置信度阈值
            
        Returns:
            dict: 检测结果
        """
        print(f"🔍 混合检测图片: {os.path.basename(image_path)}")
        
        # 1. OCR检测文本
        text_regions = self.detect_scale_text(image_path)
        
        # 2. 创建搜索区域
        search_regions = self.create_search_regions(image_path, text_regions)
        
        # 3. 在搜索区域中检测
        ocr_guided_results = self.detect_scale_in_regions(image_path, search_regions)
        print(f"   OCR引导检测到 {len(ocr_guided_results)} 个比例尺")
        
        # 4. 全图检测（备选）
        full_image_results = self.detect_scale_full_image(image_path, conf)
        print(f"   全图检测到 {len(full_image_results)} 个比例尺")
        
        # 5. 合并结果（优先OCR引导的结果）
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
        可视化检测结果
        
        Args:
            detection_result: 检测结果
            output_path: 输出图片路径
        """
        image = cv2.imread(detection_result['image_path'])
        if image is None:
            return
        
        vis_image = image.copy()
        
        # 绘制文本区域
        for region in detection_result['text_regions']:
            loc = region['location']
            x1, y1, x2, y2 = loc['left'], loc['top'], loc['left'] + loc['width'], loc['top'] + loc['height']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色：文本区域
            cv2.putText(vis_image, region['text'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 绘制搜索区域
        for region_info in detection_result['search_regions']:
            x1, y1, x2, y2 = region_info['region']
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 1)  # 黄色：搜索区域
        
        # 绘制检测结果
        for result in detection_result['final_results']:
            x1, y1, x2, y2 = map(int, result['bbox'])
            
            # 根据检测方法选择颜色
            if result['detection_method'] == 'ocr_guided':
                color = (0, 255, 0)  # 绿色：OCR引导检测
            else:
                color = (0, 0, 255)  # 红色：全图检测
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(vis_image, (x1, y1), 3, color, -1)  # 端点1
            cv2.circle(vis_image, (x2, y2), 3, color, -1)  # 端点2
            
            # 添加文本信息
            text = f"{result['text']} ({result['detection_method']})"
            cv2.putText(vis_image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"✅ 可视化结果已保存: {output_path}")
        
        return vis_image

def test_ocr_guided_detector():
    """测试OCR引导的检测器"""
    print("🧪 测试OCR引导的比例尺检测器")
    print("="*60)
    
    # 配置
    model_path = "weights/epoch80.pt"
    images_dir = "inputs/lizi/images"
    output_dir = "ocr_guided_detection"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化检测器
    detector = OCRGuidedScaleDetector(model_path)
    
    # 测试几张图片
    test_images = ['100.png', '238.png', '318.png', '159.png']
    
    for img_name in test_images:
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"❌ 图片不存在: {img_path}")
            continue
        
        print(f"\n📸 处理图片: {img_name}")
        
        # 执行混合检测
        result = detector.detect_scale_hybrid(img_path, conf=0.2)
        
        # 可视化结果
        output_path = os.path.join(output_dir, f"{img_name.replace('.png', '')}_ocr_guided.png")
        detector.visualize_results(result, output_path)
        
        print(f"   总检测数: {result['total_detections']}")
        print(f"   OCR引导检测: {len(result['ocr_guided_results'])}")
        print(f"   全图检测: {len(result['full_image_results'])}")

def test_missed_images():
    """测试之前未检测到的图片"""
    print("🧪 测试之前未检测到的图片")
    print("="*60)
    
    # 配置
    model_path = "weights/epoch80.pt"
    images_dir = "inputs/lizi/images"
    output_dir = "ocr_guided_missed_detection"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化检测器
    detector = OCRGuidedScaleDetector(model_path)
    
    # 之前未检测到的图片
    missed_images = ['87.png', '167.png', '248.png', '271.png', '298.png', '299.png', 
                    '338.png', '391.png', '393.png', '398.png', '405.png', '451.png']
    
    for img_name in missed_images:
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            print(f"❌ 图片不存在: {img_path}")
            continue
        
        print(f"\n📸 处理未检测图片: {img_name}")
        
        # 执行混合检测
        result = detector.detect_scale_hybrid(img_path, conf=0.1)  # 使用更低的置信度
        
        # 可视化结果
        output_path = os.path.join(output_dir, f"{img_name.replace('.png', '')}_ocr_guided.png")
        detector.visualize_results(result, output_path)
        
        print(f"   总检测数: {result['total_detections']}")
        print(f"   OCR引导检测: {len(result['ocr_guided_results'])}")
        print(f"   全图检测: {len(result['full_image_results'])}")

if __name__ == '__main__':
    # 测试常规图片
    test_ocr_guided_detector()
    
    # 测试之前未检测到的图片
    test_missed_images()
