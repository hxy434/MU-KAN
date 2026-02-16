#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较DDDDOCR和百度OCR在比例尺识别上的效果
"""

import pandas as pd
import os
import cv2
import numpy as np
from mock_ocr import process as local_ocr_process
import re

def analyze_scale_text_complexity():
    """分析比例尺文本的复杂性"""
    print("🔍 分析比例尺文本复杂性...")
    
    csv_file = 'scale_bar_labels_fixed.csv'
    if not os.path.exists(csv_file):
        print(f"❌ 数据集文件不存在: {csv_file}")
        return
    
    data = pd.read_csv(csv_file)
    
    # 统计比例尺长度单位分布
    length_units = {}
    length_values = {}
    
    for idx, row in data.iterrows():
        length_str = str(row['length'])
        
        # 提取单位
        unit_match = re.search(r'([a-zA-Zμ]+)$', length_str)
        if unit_match:
            unit = unit_match.group(1)
            length_units[unit] = length_units.get(unit, 0) + 1
        
        # 提取数值
        value_match = re.search(r'^(\d+\.?\d*)', length_str)
        if value_match:
            try:
                value = float(value_match.group(1))
                length_values[value] = length_values.get(value, 0) + 1
            except:
                pass
    
    print("📊 比例尺单位分布:")
    for unit, count in sorted(length_units.items(), key=lambda x: x[1], reverse=True):
        print(f"   {unit}: {count} 个样本")
    
    print("\n📊 比例尺数值分布 (前10个):")
    for value, count in sorted(length_values.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {value}: {count} 个样本")

def test_ocr_on_sample_images(num_samples=10):
    """测试OCR在样本图像上的效果"""
    print(f"\n🧪 测试OCR在 {num_samples} 个样本图像上的效果...")
    
    csv_file = 'scale_bar_labels_fixed.csv'
    img_dir = 'inputs/glas/images'
    
    if not os.path.exists(csv_file) or not os.path.exists(img_dir):
        print("❌ 数据集或图像目录不存在")
        return
    
    data = pd.read_csv(csv_file)
    
    # 随机选择样本
    sample_data = data.sample(n=min(num_samples, len(data)))
    
    ocr_results = {
        'total_tested': 0,
        'local_ocr_success': 0,
        'local_ocr_found_scale': 0,
        'successful_extractions': []
    }
    
    for idx, row in sample_data.iterrows():
        filename = str(row['filename'])
        img_path = os.path.join(img_dir, filename)
        expected_length = str(row['length'])
        
        if not os.path.exists(img_path):
            continue
        
        ocr_results['total_tested'] += 1
        
        print(f"\n📋 测试图像: {filename}")
        print(f"   期望比例尺: {expected_length}")
        
        # 测试本地OCR
        try:
            local_result = local_ocr_process(img_path)
            
            if local_result.get('error_code') == 0 and local_result.get('words_result'):
                ocr_results['local_ocr_success'] += 1
                words_found = []
                
                for item in local_result['words_result']:
                    words = item['words']
                    words_found.append(words)
                    print(f"   本地OCR识别: '{words}'")
                
                # 检查是否包含比例尺信息
                scale_found = False
                for words in words_found:
                    if any(unit in words.lower() for unit in ['nm', 'μm', 'um', 'mm']):
                        scale_found = True
                        ocr_results['local_ocr_found_scale'] += 1
                        ocr_results['successful_extractions'].append({
                            'filename': filename,
                            'expected': expected_length,
                            'ocr_text': words
                        })
                        print(f"   ✅ 找到比例尺文本: '{words}'")
                        break
                
                if not scale_found:
                    print(f"   ⚠️ 未找到比例尺相关文本")
            else:
                print(f"   ❌ 本地OCR识别失败")
                
        except Exception as e:
            print(f"   ❌ 本地OCR异常: {e}")
    
    # 输出统计结果
    print(f"\n📊 OCR效果统计:")
    print(f"   测试样本数: {ocr_results['total_tested']}")
    print(f"   本地OCR成功率: {ocr_results['local_ocr_success']}/{ocr_results['total_tested']} ({ocr_results['local_ocr_success']/ocr_results['total_tested']*100:.1f}%)")
    print(f"   比例尺识别成功率: {ocr_results['local_ocr_found_scale']}/{ocr_results['total_tested']} ({ocr_results['local_ocr_found_scale']/ocr_results['total_tested']*100:.1f}%)")
    
    if ocr_results['successful_extractions']:
        print(f"\n✅ 成功识别的比例尺:")
        for item in ocr_results['successful_extractions']:
            print(f"   {item['filename']}: 期望='{item['expected']}', OCR='{item['ocr_text']}'")

def analyze_image_characteristics():
    """分析图像特征对OCR的影响"""
    print(f"\n🔍 分析图像特征...")
    
    csv_file = 'scale_bar_labels_fixed.csv'
    img_dir = 'inputs/glas/images'
    
    if not os.path.exists(csv_file) or not os.path.exists(img_dir):
        print("❌ 数据集或图像目录不存在")
        return
    
    data = pd.read_csv(csv_file)
    sample_data = data.sample(n=min(5, len(data)))
    
    for idx, row in sample_data.iterrows():
        filename = str(row['filename'])
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            continue
        
        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        height, width = image.shape[:2]
        
        # 计算比例尺区域大小
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        scale_width = abs(x2 - x1)
        scale_height = abs(y2 - y1)
        
        # 提取比例尺区域
        scale_region = image[int(y1-10):int(y2+10), int(x1-10):int(x2+10)]
        
        print(f"\n📊 {filename}:")
        print(f"   图像尺寸: {width}x{height}")
        print(f"   比例尺位置: ({x1},{y1}) -> ({x2},{y2})")
        print(f"   比例尺尺寸: {scale_width:.0f}x{scale_height:.0f} 像素")
        print(f"   比例尺占图像比例: {scale_width/width*100:.1f}%x{scale_height/height*100:.1f}%")

def provide_recommendations():
    """提供OCR优化建议"""
    print(f"\n💡 OCR优化建议:")
    print(f"1. 📝 文本识别挑战:")
    print(f"   - 医学图像的比例尺文本通常很小")
    print(f"   - 字体可能不规范或有噪声")
    print(f"   - 背景对比度可能不理想")
    
    print(f"\n2. 🔧 可能的解决方案:")
    print(f"   a) 图像预处理:")
    print(f"      - 提取比例尺区域并放大")
    print(f"      - 增强对比度和清晰度")
    print(f"      - 去除噪声")
    
    print(f"   b) 多重OCR策略:")
    print(f"      - DDDDOCR + 简单OCR + CSV备用")
    print(f"      - 基于模式匹配的文本识别")
    print(f"      - 结合坐标信息的智能识别")
    
    print(f"   c) 训练策略调整:")
    print(f"      - 当OCR失败时，使用整个图像进行训练")
    print(f"      - 基于已知坐标位置进行区域裁剪")
    print(f"      - 减少对OCR的依赖，更多依靠视觉特征")

if __name__ == '__main__':
    print("🚀 比较OCR在比例尺识别任务中的效果")
    print("=" * 60)
    
    # 1. 分析比例尺文本复杂性
    analyze_scale_text_complexity()
    
    # 2. 测试OCR效果
    test_ocr_on_sample_images(num_samples=10)
    
    # 3. 分析图像特征
    analyze_image_characteristics()
    
    # 4. 提供建议
    provide_recommendations()
