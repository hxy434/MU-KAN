#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算真实的相对误差
使用scale_bar_labels_fixed.csv中的真实坐标
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_real_error():
    """计算真实的相对误差"""
    
    print("🔍 开始计算真实的相对误差...")
    
    # 读取预测结果
    pred_df = pd.read_csv('scale_endpoints_data.csv')
    print(f"📊 预测数据: {len(pred_df)} 条记录")
    
    # 读取真实标签
    gt_df = pd.read_csv('scale_bar_labels_fixed.csv')
    print(f"📊 真实标签: {len(gt_df)} 条记录")
    
    # 计算真实标签的长度
    gt_df['true_length'] = np.sqrt((gt_df['x2'] - gt_df['x1'])**2 + (gt_df['y2'] - gt_df['y1'])**2)
    
    # 创建结果列表
    error_results = []
    
    for idx, pred_row in pred_df.iterrows():
        image_name = pred_row['image_name']
        
        # 在真实标签中查找对应的图片
        # 预测文件中的图片名称是数字，真实标签中是带.png后缀的
        gt_match = gt_df[gt_df['filename'] == f"{int(image_name)}.png"]
        
        if len(gt_match) > 0:
            gt_row = gt_match.iloc[0]
            true_length = gt_row['true_length']
            pred_length = pred_row['predicted_length']
            
            # 计算相对误差
            if true_length > 0:
                relative_error = abs(pred_length - true_length) / true_length * 100
            else:
                relative_error = 0
            
            error_results.append({
                'image_name': image_name,
                'predicted_length': pred_length,
                'true_length': true_length,
                'relative_error_percent': relative_error,
                'absolute_error': abs(pred_length - true_length),
                'pred_endpoint1': (pred_row['endpoint1_x'], pred_row['endpoint1_y']),
                'pred_endpoint2': (pred_row['endpoint2_x'], pred_row['endpoint2_y']),
                'true_coords': (gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2'])
            })
        else:
            print(f"⚠️ 未找到图片 {image_name} 的真实标签")
    
    # 转换为DataFrame
    error_df = pd.DataFrame(error_results)
    
    if len(error_df) == 0:
        print("❌ 没有找到匹配的数据")
        return
    
    print(f"✅ 成功匹配: {len(error_df)} 条记录")
    
    # 计算统计信息
    avg_error = error_df['relative_error_percent'].mean()
    median_error = error_df['relative_error_percent'].median()
    max_error = error_df['relative_error_percent'].max()
    min_error = error_df['relative_error_percent'].min()
    std_error = error_df['relative_error_percent'].std()
    
    # 误差分布统计
    error_under_5 = len(error_df[error_df['relative_error_percent'] <= 5])
    error_under_10 = len(error_df[error_df['relative_error_percent'] <= 10])
    error_under_20 = len(error_df[error_df['relative_error_percent'] <= 20])
    
    print("\n🎯 真实相对误差统计报告")
    print("="*60)
    print(f"📊 基础统计:")
    print(f"   • 有效样本数: {len(error_df)}")
    print(f"   • 平均相对误差: {avg_error:.2f}%")
    print(f"   • 中位相对误差: {median_error:.2f}%")
    print(f"   • 最大相对误差: {max_error:.2f}%")
    print(f"   • 最小相对误差: {min_error:.2f}%")
    print(f"   • 误差标准差: {std_error:.2f}%")
    print("")
    print(f"🎯 误差分布:")
    print(f"   • ≤5% 误差: {error_under_5}/{len(error_df)} ({error_under_5/len(error_df)*100:.1f}%)")
    print(f"   • ≤10% 误差: {error_under_10}/{len(error_df)} ({error_under_10/len(error_df)*100:.1f}%)")
    print(f"   • ≤20% 误差: {error_under_20}/{len(error_df)} ({error_under_20/len(error_df)*100:.1f}%)")
    print("="*60)
    
    # 保存详细结果
    output_file = 'real_error_analysis.csv'
    error_df.to_csv(output_file, index=False)
    print(f"📄 详细误差分析保存: {output_file}")
    
    # 创建可视化
    create_error_visualization(error_df)
    
    # 按误差从高到低排序
    error_df_sorted = error_df.sort_values('relative_error_percent', ascending=False)
    
    print("\n📊 误差从高到低排序 (前20名):")
    print("="*90)
    print(f"{'排名':<6} {'图片名':<10} {'预测长度':<12} {'真实长度':<12} {'相对误差':<12} {'绝对误差':<12}")
    print("-"*90)
    
    for i, (idx, row) in enumerate(error_df_sorted.head(20).iterrows(), 1):
        print(f"{i:<6} {row['image_name']:<10} {row['predicted_length']:<12.1f} {row['true_length']:<12.1f} "
              f"{row['relative_error_percent']:<12.2f}% {row['absolute_error']:<12.1f}")
    
    print("-"*90)
    
    # 显示最佳和最差的结果
    print("\n🏆 最佳预测 (误差最小):")
    best_idx = error_df['relative_error_percent'].idxmin()
    best_row = error_df.loc[best_idx]
    print(f"   图片: {best_row['image_name']}")
    print(f"   预测长度: {best_row['predicted_length']:.1f}px")
    print(f"   真实长度: {best_row['true_length']:.1f}px")
    print(f"   相对误差: {best_row['relative_error_percent']:.2f}%")
    
    print("\n❌ 最差预测 (误差最大):")
    worst_idx = error_df['relative_error_percent'].idxmax()
    worst_row = error_df.loc[worst_idx]
    print(f"   图片: {worst_row['image_name']}")
    print(f"   预测长度: {worst_row['predicted_length']:.1f}px")
    print(f"   真实长度: {worst_row['true_length']:.1f}px")
    print(f"   相对误差: {worst_row['relative_error_percent']:.2f}%")
    
    # 保存排序后的结果
    sorted_output_file = 'real_error_analysis_sorted.csv'
    error_df_sorted.to_csv(sorted_output_file, index=False)
    print(f"\n📄 排序后的误差分析保存: {sorted_output_file}")
    
    # 列出相对误差最大的十个文件
    print("\n🔴 相对误差最大的十张图片:")
    print("="*90)
    print(f"{'排名':<6} {'图片名':<10} {'预测长度':<12} {'真实长度':<12} {'相对误差':<12} {'绝对误差':<12}")
    print("-"*90)
    
    # 按相对误差从高到低排序，找出误差最大的十张图片
    largest_error_sorted = error_df.sort_values('relative_error_percent', ascending=False)
    
    for i, (idx, row) in enumerate(largest_error_sorted.head(10).iterrows(), 1):
        print(f"{i:<6} {row['image_name']:<10} {row['predicted_length']:<12.1f} {row['true_length']:<12.1f} "
              f"{row['relative_error_percent']:<12.2f}% {row['absolute_error']:<12.1f}")
    
    print("-"*90)
    print(f"📊 最大误差: {largest_error_sorted['relative_error_percent'].max():.2f}%")
    print(f"📊 最小误差: {largest_error_sorted['relative_error_percent'].min():.2f}%")
    
    # 保存最大误差分析
    largest_error_output_file = 'largest_error_analysis.csv'
    largest_error_sorted.to_csv(largest_error_output_file, index=False)
    print(f"📄 最大误差分析保存: {largest_error_output_file}")
    
    # 列出相对误差相差最大的十个文件（与平均误差的差值）
    print("\n🔥 相对误差相差最大的十个文件（与平均误差的差值）:")
    print("="*90)
    print(f"{'排名':<6} {'图片名':<10} {'预测长度':<12} {'真实长度':<12} {'相对误差':<12} {'误差差值':<12}")
    print("-"*90)
    
    # 计算误差差值（与平均误差的差值）
    mean_error = error_df['relative_error_percent'].mean()
    error_df_sorted['error_diff'] = abs(error_df_sorted['relative_error_percent'] - mean_error)
    
    # 按误差差值排序，找出与平均误差相差最大的
    error_diff_sorted = error_df_sorted.sort_values('error_diff', ascending=False)
    
    for i, (idx, row) in enumerate(error_diff_sorted.head(10).iterrows(), 1):
        error_diff = row['error_diff']
        print(f"{i:<6} {row['image_name']:<10} {row['predicted_length']:<12.1f} {row['true_length']:<12.1f} "
              f"{row['relative_error_percent']:<12.2f}% {error_diff:<12.2f}%")
    
    print("-"*90)
    print(f"📊 平均误差: {mean_error:.2f}%")
    print(f"📊 误差差值范围: {error_diff_sorted['error_diff'].min():.2f}% - {error_diff_sorted['error_diff'].max():.2f}%")
    
    # 保存误差差值分析
    error_diff_output_file = 'error_difference_analysis.csv'
    error_diff_sorted.to_csv(error_diff_output_file, index=False)
    print(f"📄 误差差值分析保存: {error_diff_output_file}")
    
    return error_df

def create_error_visualization(error_df):
    """创建误差可视化图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('比例尺长度预测误差分析', fontsize=16, fontweight='bold')
    
    # 子图1: 预测vs真实长度散点图
    ax1.scatter(error_df['true_length'], error_df['predicted_length'], alpha=0.6, color='blue')
    ax1.plot([error_df['true_length'].min(), error_df['true_length'].max()], 
             [error_df['true_length'].min(), error_df['true_length'].max()], 
             'r--', label='完美预测线')
    ax1.set_xlabel('真实长度 (像素)')
    ax1.set_ylabel('预测长度 (像素)')
    ax1.set_title('预测长度 vs 真实长度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 相对误差分布直方图
    ax2.hist(error_df['relative_error_percent'], bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(error_df['relative_error_percent'].mean(), color='red', linestyle='--', 
                label=f'平均误差: {error_df["relative_error_percent"].mean():.1f}%')
    ax2.set_xlabel('相对误差 (%)')
    ax2.set_ylabel('频次')
    ax2.set_title('相对误差分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 相对误差按图片排序
    image_indices = range(1, len(error_df) + 1)
    ax3.plot(image_indices, error_df['relative_error_percent'], 'b-o', markersize=4)
    ax3.axhline(y=5, color='orange', linestyle='--', label='5%误差线', alpha=0.7)
    ax3.axhline(y=10, color='red', linestyle='--', label='10%误差线', alpha=0.7)
    ax3.set_xlabel('图片序号')
    ax3.set_ylabel('相对误差 (%)')
    ax3.set_title('相对误差变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 误差分布饼图
    error_ranges = [
        len(error_df[error_df['relative_error_percent'] <= 5]),
        len(error_df[(error_df['relative_error_percent'] > 5) & (error_df['relative_error_percent'] <= 10)]),
        len(error_df[(error_df['relative_error_percent'] > 10) & (error_df['relative_error_percent'] <= 20)]),
        len(error_df[error_df['relative_error_percent'] > 20])
    ]
    labels = ['≤5%', '5-10%', '10-20%', '>20%']
    colors = ['green', 'yellow', 'orange', 'red']
    
    ax4.pie(error_ranges, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('误差分布比例')
    
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('real_error_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 误差分析图表保存: real_error_analysis.png")
    plt.show()

if __name__ == '__main__':
    calculate_real_error()
