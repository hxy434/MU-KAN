#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算最佳模型的比例尺长度误差
"""

import os
import pandas as pd
import torch
from ultralytics import YOLO

def calculate_scale_length_error(model_path, gt_csv='scale_bar_labels_fixed.csv', val_images_dir='yolo11_optimized_dataset/images/val'):
    """计算比例尺长度误差"""
    try:
        # 读取真实标签
        gt_df = pd.read_csv(gt_csv)
        gt_df['true_length'] = abs(gt_df['x2'] - gt_df['x1'])  # 使用横坐标差值
        
        # 检查验证集路径
        if not os.path.exists(val_images_dir):
            print(f"❌ 验证集路径不存在: {val_images_dir}")
            return float('inf')
        
        # 加载模型
        print(f"🔧 加载模型: {model_path}")
        model = YOLO(model_path)
        
        total_error = 0
        valid_predictions = 0
        
        # 在验证集上预测
        print("🔍 开始计算比例尺长度误差...")
        for img_file in os.listdir(val_images_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(val_images_dir, img_file)
            results = model.predict(img_path, conf=0.5, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                # 获取预测的边界框
                pred_box = results[0].boxes.xyxy[0].cpu().numpy()
                pred_length = abs(pred_box[2] - pred_box[0])  # 使用横坐标差值
                
                # 查找对应的真实标签
                img_name = img_file
                gt_match = gt_df[gt_df['filename'] == img_name]
                
                if len(gt_match) > 0:
                    true_length = gt_match.iloc[0]['true_length']
                    if true_length > 0:
                        relative_error = abs(pred_length - true_length) / true_length * 100
                        total_error += relative_error
                        valid_predictions += 1
                        print(f"   {img_file}: 预测={pred_length:.1f}px, 真实={true_length:.1f}px, 误差={relative_error:.2f}%")
        
        # 返回平均相对误差
        if valid_predictions > 0:
            avg_error = total_error / valid_predictions
            print(f"\n📊 统计结果:")
            print(f"   有效预测数: {valid_predictions}")
            print(f"   平均相对误差: {avg_error:.2f}%")
            return avg_error
        else:
            print("❌ 没有有效的预测结果")
            return float('inf')
            
    except Exception as e:
        print(f"❌ 计算长度误差时出错: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

def main():
    """主函数"""
    print("🎯 计算最佳模型的比例尺长度误差")
    print("="*50)
    
    # 最佳模型路径
    best_model_path = "yolo11_optimized_runs/optimized_scalebar_detection3/weights/best.pt"
    
    if not os.path.exists(best_model_path):
        print(f"❌ 最佳模型文件不存在: {best_model_path}")
        return
    
    # 计算误差
    error = calculate_scale_length_error(best_model_path)
    
    print(f"\n🎯 最终结果:")
    print(f"   模型: {best_model_path}")
    print(f"   比例尺长度误差: {error:.2f}%")
    
    if error <= 5.0:
        print(f"🎉 优秀! 误差: {error:.2f}% <= 5%")
    elif error <= 10.0:
        print(f"✅ 良好! 误差: {error:.2f}% <= 10%")
    else:
        print(f"⚠️ 需要进一步优化，当前误差: {error:.2f}%")

if __name__ == '__main__':
    main()
