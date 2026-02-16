#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析比例尺数据分布和特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def analyze_scale_data():
    """分析比例尺数据"""
    print("🔍 分析比例尺数据分布")
    print("="*50)
    
    # 读取数据
    df = pd.read_csv('scale_bar_labels_fixed.csv')
    
    # 计算比例尺长度
    df['scale_length'] = abs(df['x2'] - df['x1'])
    
    print(f"📊 数据统计:")
    print(f"   总样本数: {len(df)}")
    print(f"   平均长度: {df['scale_length'].mean():.2f} 像素")
    print(f"   标准差: {df['scale_length'].std():.2f} 像素")
    print(f"   最小长度: {df['scale_length'].min():.0f} 像素")
    print(f"   最大长度: {df['scale_length'].max():.0f} 像素")
    print(f"   中位数: {df['scale_length'].median():.2f} 像素")
    
    # 长度分布
    print(f"\n📏 长度分布:")
    length_ranges = [
        (0, 50, "0-50像素"),
        (50, 100, "50-100像素"),
        (100, 200, "100-200像素"),
        (200, 500, "200-500像素"),
        (500, 1000, "500-1000像素"),
        (1000, float('inf'), "1000+像素")
    ]
    
    for min_len, max_len, label in length_ranges:
        if max_len == float('inf'):
            count = len(df[df['scale_length'] >= min_len])
        else:
            count = len(df[(df['scale_length'] >= min_len) & (df['scale_length'] < max_len)])
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} 个 ({percentage:.1f}%)")
    
    # 分析图像尺寸
    print(f"\n🖼️ 图像尺寸分析:")
    img_dir = 'inputs/lizi/images'
    sample_images = df['filename'].head(20).tolist()
    
    widths = []
    heights = []
    
    for filename in sample_images:
        img_path = os.path.join(img_dir, str(filename))
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
    
    if widths:
        print(f"   平均宽度: {np.mean(widths):.0f} 像素")
        print(f"   平均高度: {np.mean(heights):.0f} 像素")
        print(f"   宽度范围: {min(widths)} - {max(widths)} 像素")
        print(f"   高度范围: {min(heights)} - {max(heights)} 像素")
    
    # 分析坐标分布
    print(f"\n📍 坐标分布:")
    print(f"   x1范围: {df['x1'].min():.0f} - {df['x1'].max():.0f}")
    print(f"   y1范围: {df['y1'].min():.0f} - {df['y1'].max():.0f}")
    print(f"   x2范围: {df['x2'].min():.0f} - {df['x2'].max():.0f}")
    print(f"   y2范围: {df['y2'].min():.0f} - {df['y2'].max():.0f}")
    
    # 检查异常值
    print(f"\n⚠️ 异常值检查:")
    # 检查长度为0的情况
    zero_length = df[df['scale_length'] == 0]
    if len(zero_length) > 0:
        print(f"   长度为0的样本: {len(zero_length)} 个")
        print(f"   文件: {zero_length['filename'].tolist()}")
    
    # 检查极短比例尺
    very_short = df[df['scale_length'] < 10]
    if len(very_short) > 0:
        print(f"   极短比例尺(<10像素): {len(very_short)} 个")
    
    # 检查极长比例尺
    very_long = df[df['scale_length'] > 500]
    if len(very_long) > 0:
        print(f"   极长比例尺(>500像素): {len(very_long)} 个")
    
    # 绘制长度分布直方图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['scale_length'], bins=50, alpha=0.7, color='blue')
    plt.title('比例尺长度分布')
    plt.xlabel('长度 (像素)')
    plt.ylabel('频次')
    plt.yscale('log')
    
    plt.subplot(2, 2, 2)
    plt.hist(df['scale_length'], bins=50, alpha=0.7, color='red')
    plt.title('比例尺长度分布 (线性)')
    plt.xlabel('长度 (像素)')
    plt.ylabel('频次')
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['x1'], df['y1'], alpha=0.5, s=1)
    plt.title('起点坐标分布')
    plt.xlabel('x1')
    plt.ylabel('y1')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['x2'], df['y2'], alpha=0.5, s=1)
    plt.title('终点坐标分布')
    plt.xlabel('x2')
    plt.ylabel('y2')
    
    plt.tight_layout()
    plt.savefig('scale_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def suggest_improvements(df):
    """根据数据分析结果提出改进建议"""
    print(f"\n💡 改进建议:")
    print("="*50)
    
    # 计算长度统计
    lengths = df['scale_length'].values
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    print(f"1. 数据标准化:")
    print(f"   - 当前长度范围: {lengths.min():.0f} - {lengths.max():.0f} 像素")
    print(f"   - 建议使用对数变换: log(length + 1)")
    print(f"   - 或者使用标准化: (length - mean) / std")
    
    print(f"\n2. 数据过滤:")
    print(f"   - 移除长度为0的异常样本")
    print(f"   - 考虑移除极短(<5像素)和极长(>1000像素)的样本")
    
    print(f"\n3. 模型改进:")
    print(f"   - 使用对数损失函数")
    print(f"   - 添加BatchNorm和Dropout")
    print(f"   - 使用更小的学习率")
    
    print(f"\n4. 数据增强:")
    print(f"   - 减少数据增强，保持比例尺特征")
    print(f"   - 使用固定尺寸输入")
    
    # 计算对数变换后的统计
    log_lengths = np.log1p(lengths)
    print(f"\n5. 对数变换效果:")
    print(f"   - 原始长度标准差: {std_length:.2f}")
    print(f"   - 对数长度标准差: {np.std(log_lengths):.2f}")
    print(f"   - 对数长度范围: {log_lengths.min():.2f} - {log_lengths.max():.2f}")

if __name__ == '__main__':
    df = analyze_scale_data()
    suggest_improvements(df)
