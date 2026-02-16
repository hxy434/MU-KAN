#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze scale bar data distribution and features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def analyze_scale_data():
    """Analyze scale bar data"""
    print(" Analyzing scale bar data distribution")
    print("="*50)
    
    # Read data
    df = pd.read_csv('scale_bar_labels_fixed.csv')
    
    # Calculate scale bar length
    df['scale_length'] = abs(df['x2'] - df['x1'])
    
    print(f"   Data Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Average length: {df['scale_length'].mean():.2f} pixels")
    print(f"   Standard deviation: {df['scale_length'].std():.2f} pixels")
    print(f"   Minimum length: {df['scale_length'].min():.0f} pixels")
    print(f"   Maximum length: {df['scale_length'].max():.0f} pixels")
    print(f"   Median: {df['scale_length'].median():.2f} pixels")
    
    # Length distribution
    print(f"\n Length Distribution:")
    length_ranges = [
        (0, 50, "0-50 pixels"),
        (50, 100, "50-100 pixels"),
        (100, 200, "100-200 pixels"),
        (200, 500, "200-500 pixels"),
        (500, 1000, "500-1000 pixels"),
        (1000, float('inf'), "1000+ pixels")
    ]
    
    for min_len, max_len, label in length_ranges:
        if max_len == float('inf'):
            count = len(df[df['scale_length'] >= min_len])
        else:
            count = len(df[(df['scale_length'] >= min_len) & (df['scale_length'] < max_len)])
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} samples ({percentage:.1f}%)")
    
    # Analyze image dimensions
    print(f"\n Image Dimension Analysis:")
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
        print(f"   Average width: {np.mean(widths):.0f} pixels")
        print(f"   Average height: {np.mean(heights):.0f} pixels")
        print(f"   Width range: {min(widths)} - {max(widths)} pixels")
        print(f"   Height range: {min(heights)} - {max(heights)} pixels")
    
    # Analyze coordinate distribution
    print(f"\n Coordinate Distribution:")
    print(f"   x1 range: {df['x1'].min():.0f} - {df['x1'].max():.0f}")
    print(f"   y1 range: {df['y1'].min():.0f} - {df['y1'].max():.0f}")
    print(f"   x2 range: {df['x2'].min():.0f} - {df['x2'].max():.0f}")
    print(f"   y2 range: {df['y2'].min():.0f} - {df['y2'].max():.0f}")
    
    # Check outliers
    print(f"\n Outlier Check:")
    # Check zero length cases
    zero_length = df[df['scale_length'] == 0]
    if len(zero_length) > 0:
        print(f"   Samples with zero length: {len(zero_length)}")
        print(f"   Files: {zero_length['filename'].tolist()}")
    
    # Check extremely short scale bars
    very_short = df[df['scale_length'] < 10]
    if len(very_short) > 0:
        print(f"   Extremely short scale bars (<10 pixels): {len(very_short)} samples")
    
    # Check extremely long scale bars
    very_long = df[df['scale_length'] > 500]
    if len(very_long) > 0:
        print(f"   Extremely long scale bars (>500 pixels): {len(very_long)} samples")
    
    # Plot length distribution histogram
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(df['scale_length'], bins=50, alpha=0.7, color='blue')
    plt.title('Scale Bar Length Distribution')
    plt.xlabel('Length (pixels)')
    plt.ylabel('Frequency')
    plt.yscale('log')
    
    plt.subplot(2, 2, 2)
    plt.hist(df['scale_length'], bins=50, alpha=0.7, color='red')
    plt.title('Scale Bar Length Distribution (Linear)')
    plt.xlabel('Length (pixels)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['x1'], df['y1'], alpha=0.5, s=1)
    plt.title('Start Coordinate Distribution')
    plt.xlabel('x1')
    plt.ylabel('y1')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['x2'], df['y2'], alpha=0.5, s=1)
    plt.title('End Coordinate Distribution')
    plt.xlabel('x2')
    plt.ylabel('y2')
    
    plt.tight_layout()
    plt.savefig('scale_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df

def suggest_improvements(df):
    """Propose improvement suggestions based on data analysis results"""
    print(f"\n Improvement Suggestions:")
    print("="*50)
    
    # Calculate length statistics
    lengths = df['scale_length'].values
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    print(f"1. Data Standardization:")
    print(f"   - Current length range: {lengths.min():.0f} - {lengths.max():.0f} pixels")
    print(f"   - Suggest using log transformation: log(length + 1)")
    print(f"   - Or use standardization: (length - mean) / std")
    
    print(f"\n2. Data Filtering:")
    print(f"   - Remove abnormal samples with zero length")
    print(f"   - Consider removing extremely short (<5 pixels) and extremely long (>1000 pixels) samples")
    
    print(f"\n3. Model Improvement:")
    print(f"   - Use logarithmic loss function")
    print(f"   - Add BatchNorm and Dropout")
    print(f"   - Use smaller learning rate")
    
    print(f"\n4. Data Augmentation:")
    print(f"   - Reduce data augmentation to preserve scale bar features")
    print(f"   - Use fixed-size inputs")
    
    # Calculate statistics after log transformation
    log_lengths = np.log1p(lengths)
    print(f"\n5. Log Transformation Effect:")
    print(f"   - Original length standard deviation: {std_length:.2f}")
    print(f"   - Log length standard deviation: {np.std(log_lengths):.2f}")
    print(f"   - Log length range: {log_lengths.min():.2f} - {log_lengths.max():.2f}")

if __name__ == '__main__':
    df = analyze_scale_data()
    suggest_improvements(df)
