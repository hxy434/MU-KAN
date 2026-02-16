#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理CSV文件，删除检测失败的记录
"""

import pandas as pd
import os

def clean_csv_results():
    """清理CSV文件，删除检测失败的记录"""
    print("🧹 清理CSV文件，删除检测失败的记录")
    print("="*50)
    
    # 读取原始CSV文件
    input_csv = "epoch80_test_results.csv"
    if not os.path.exists(input_csv):
        print(f"❌ 文件不存在: {input_csv}")
        return
    
    # 读取数据
    df = pd.read_csv(input_csv)
    print(f"📊 原始数据: {len(df)} 条记录")
    
    # 统计检测结果
    detected_count = len(df[df['detected'] == True])
    undetected_count = len(df[df['detected'] == False])
    
    print(f"✅ 成功检测: {detected_count} 条")
    print(f"❌ 检测失败: {undetected_count} 条")
    
    # 只保留成功检测的记录
    cleaned_df = df[df['detected'] == True].copy()
    
    print(f"🧹 清理后数据: {len(cleaned_df)} 条记录")
    
    # 保存清理后的文件
    output_csv = "epoch80_test_results_cleaned.csv"
    cleaned_df.to_csv(output_csv, index=False)
    print(f"✅ 清理后的文件已保存: {output_csv}")
    
    # 重新计算统计信息
    if len(cleaned_df) > 0:
        avg_relative_error = cleaned_df['relative_error'].mean()
        avg_absolute_error = cleaned_df['absolute_error'].mean()
        min_error = cleaned_df['relative_error'].min()
        max_error = cleaned_df['relative_error'].max()
        
        print(f"\n📊 清理后的统计信息:")
        print(f"平均相对误差: {avg_relative_error:.2f}%")
        print(f"平均绝对误差: {avg_absolute_error:.1f} 像素")
        print(f"最小相对误差: {min_error:.2f}%")
        print(f"最大相对误差: {max_error:.2f}%")
        
        # 误差分布
        error_ranges = [
            (0, 5, "0-5%"),
            (5, 10, "5-10%"),
            (10, 20, "10-20%"),
            (20, 50, "20-50%"),
            (50, float('inf'), ">50%")
        ]
        
        print(f"\n误差分布:")
        for min_err, max_err, label in error_ranges:
            if max_err == float('inf'):
                count = len(cleaned_df[cleaned_df['relative_error'] >= min_err])
            else:
                count = len(cleaned_df[(cleaned_df['relative_error'] >= min_err) & (cleaned_df['relative_error'] < max_err)])
            percentage = count / len(cleaned_df) * 100
            print(f"  {label}: {count} 张 ({percentage:.1f}%)")
    
    # 更新统计文件
    stats_csv = "epoch80_test_stats_cleaned.csv"
    stats_data = {
        'metric': [
            'total_images',
            'detected_count',
            'undetected_count',
            'detection_rate',
            'avg_relative_error',
            'avg_absolute_error',
            'min_relative_error',
            'max_relative_error'
        ],
        'value': [
            len(df),  # 原始总图片数
            len(cleaned_df),  # 成功检测数
            undetected_count,  # 未检测数
            len(cleaned_df)/len(df)*100 if len(df) > 0 else 0,  # 检测率
            cleaned_df['relative_error'].mean() if len(cleaned_df) > 0 else None,
            cleaned_df['absolute_error'].mean() if len(cleaned_df) > 0 else None,
            cleaned_df['relative_error'].min() if len(cleaned_df) > 0 else None,
            cleaned_df['relative_error'].max() if len(cleaned_df) > 0 else None
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(stats_csv, index=False)
    print(f"✅ 更新后的统计信息已保存: {stats_csv}")
    
    return cleaned_df

def show_cleaned_summary():
    """显示清理后的摘要信息"""
    print("\n" + "="*50)
    print("📋 清理摘要")
    print("="*50)
    
    # 检查清理后的文件
    cleaned_csv = "epoch80_test_results_cleaned.csv"
    if os.path.exists(cleaned_csv):
        df = pd.read_csv(cleaned_csv)
        print(f"✅ 清理后的文件包含 {len(df)} 条成功检测的记录")
        print(f"📁 文件路径: {cleaned_csv}")
        
        # 显示前几行
        print(f"\n📄 前5条记录预览:")
        print(df.head().to_string(index=False))
    else:
        print("❌ 清理后的文件不存在")

if __name__ == '__main__':
    # 清理CSV文件
    cleaned_df = clean_csv_results()
    
    # 显示摘要
    show_cleaned_summary()
