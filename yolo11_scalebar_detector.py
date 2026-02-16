#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11比例尺检测器 - 最新最强版本
使用YOLOv11进行高精度比例尺检测
"""

import os
import cv2
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
import sys

class YOLO11ScaleBarConverter:
    """转换数据格式为YOLOv11格式"""
    
    def __init__(self, csv_file, img_dir, output_dir="yolo11_dataset"):
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.data = pd.read_csv(csv_file)
        
        # 创建YOLOv11目录结构
        self.setup_directories()
    
    def setup_directories(self):
        """创建YOLOv11目录结构"""
        dirs = [
            f"{self.output_dir}/images/train",
            f"{self.output_dir}/images/val", 
            f"{self.output_dir}/images/test",
            f"{self.output_dir}/labels/train",
            f"{self.output_dir}/labels/val",
            f"{self.output_dir}/labels/test"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"✅ YOLOv11目录结构创建完成: {self.output_dir}")
    
    def convert_to_yolo_format(self):
        """转换为YOLOv11格式 - 优化的边界框生成"""
        print("🔄 转换数据为YOLOv11格式...")
        
        valid_samples = []
        
        # 验证和转换数据
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="处理数据"):
            img_path = os.path.join(self.img_dir, str(row['filename']))
            
            if not os.path.exists(img_path):
                continue
                
            # 加载图片获取尺寸
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            h, w = image.shape[:2]
            
            # 验证坐标
            if not (0 <= row['x1'] < w and 0 <= row['x2'] < w and
                    0 <= row['y1'] < h and 0 <= row['y2'] < h):
                continue
            
            # 计算YOLOv11格式的边界框 - 更智能的方法
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            
            # 计算比例尺的角度和长度
            line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            # 自适应边界框尺寸
            # 长度方向：比例尺长度 + 小边距
            # 宽度方向：自适应厚度
            length_margin = max(5, line_length * 0.05)  # 5%边距，最小5像素
            width_thickness = max(8, min(w, h) // 80)   # 自适应厚度
            
            # 计算旋转边界框的四个顶点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 扩展长度
            half_length = (line_length + 2 * length_margin) / 2
            half_width = width_thickness / 2
            
            # 旋转后的边界框顶点
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            corners = [
                (-half_length, -half_width),
                (half_length, -half_width),
                (half_length, half_width),
                (-half_length, half_width)
            ]
            
            rotated_corners = []
            for cx, cy in corners:
                rx = cx * cos_a - cy * sin_a + center_x
                ry = cx * sin_a + cy * cos_a + center_y
                rotated_corners.append((rx, ry))
            
            # 计算轴对齐边界框
            xs = [c[0] for c in rotated_corners]
            ys = [c[1] for c in rotated_corners]
            
            bbox_x1 = max(0, min(xs))
            bbox_y1 = max(0, min(ys))
            bbox_x2 = min(w, max(xs))
            bbox_y2 = min(h, max(ys))
            
            # 确保边界框有效
            if bbox_x2 <= bbox_x1 or bbox_y2 <= bbox_y1:
                continue
            
            # 转换为YOLOv11格式 (中心点坐标 + 宽高，归一化)
            center_x_norm = (bbox_x1 + bbox_x2) / 2 / w
            center_y_norm = (bbox_y1 + bbox_y2) / 2 / h
            width_norm = (bbox_x2 - bbox_x1) / w
            height_norm = (bbox_y2 - bbox_y1) / h
            
            # 添加质量检查
            if (0.001 < width_norm < 0.9 and 0.001 < height_norm < 0.9 and
                0.001 < center_x_norm < 0.999 and 0.001 < center_y_norm < 0.999):
                
                valid_samples.append({
                    'filename': row['filename'],
                    'img_path': img_path,
                    'yolo_label': f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}",
                    'original_coords': (x1, y1, x2, y2),
                    'bbox_coords': (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                    'line_length': line_length,
                    'angle': angle
                })
        
        print(f"✅ 有效样本: {len(valid_samples)}")
        
        # 数据集划分 - 更好的划分策略
        np.random.seed(42)  # 固定随机种子
        np.random.shuffle(valid_samples)
        
        # 按长度分层划分，确保各长度范围都有代表性
        lengths = [s['line_length'] for s in valid_samples]
        length_percentiles = np.percentile(lengths, [33, 66])
        
        short_samples = [s for s in valid_samples if s['line_length'] <= length_percentiles[0]]
        medium_samples = [s for s in valid_samples if length_percentiles[0] < s['line_length'] <= length_percentiles[1]]
        long_samples = [s for s in valid_samples if s['line_length'] > length_percentiles[1]]
        
        def split_samples(samples, train_ratio=0.7, val_ratio=0.2):
            n = len(samples)
            train_n = int(n * train_ratio)
            val_n = int(n * val_ratio)
            
            return samples[:train_n], samples[train_n:train_n+val_n], samples[train_n+val_n:]
        
        # 分层划分
        short_train, short_val, short_test = split_samples(short_samples)
        medium_train, medium_val, medium_test = split_samples(medium_samples)
        long_train, long_val, long_test = split_samples(long_samples)
        
        # 合并
        train_samples = short_train + medium_train + long_train
        val_samples = short_val + medium_val + long_val
        test_samples = short_test + medium_test + long_test
        
        print(f"📊 分层数据划分:")
        print(f"   短比例尺: {len(short_samples)} 个")
        print(f"   中比例尺: {len(medium_samples)} 个")
        print(f"   长比例尺: {len(long_samples)} 个")
        print(f"   训练集: {len(train_samples)}, 验证集: {len(val_samples)}, 测试集: {len(test_samples)}")
        
        # 复制文件和创建标签
        self._copy_dataset(train_samples, "train")
        self._copy_dataset(val_samples, "val")
        self._copy_dataset(test_samples, "test")
        
        # 创建YOLOv11 YAML配置文件
        self._create_yaml_config()
        
        return len(valid_samples)
    
    def _copy_dataset(self, samples, split):
        """复制数据集文件"""
        for sample in tqdm(samples, desc=f"复制{split}数据"):
            # 复制图片
            src_img = sample['img_path']
            dst_img = f"{self.output_dir}/images/{split}/{sample['filename']}"
            shutil.copy2(src_img, dst_img)
            
            # 创建标签文件
            label_file = f"{self.output_dir}/labels/{split}/{Path(sample['filename']).stem}.txt"
            with open(label_file, 'w') as f:
                f.write(sample['yolo_label'] + '\n')
    
    def _create_yaml_config(self):
        """创建YOLOv11配置文件"""
        config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # 类别数量
            'names': ['scalebar']  # 类别名称
        }
        
        yaml_path = f"{self.output_dir}/scalebar_v11.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ YOLOv11配置文件创建: {yaml_path}")
        return yaml_path

class YOLO11ScaleBarTrainer:
    """YOLOv11比例尺检测器训练器"""
    
    def __init__(self, dataset_dir="yolo11_dataset"):
        self.dataset_dir = dataset_dir
        self.yaml_path = f"{dataset_dir}/scalebar_v11.yaml"
        
    def install_ultralytics(self):
        """安装最新版ultralytics (支持YOLOv11)"""
        try:
            import ultralytics
            from ultralytics import YOLO
            
            # 检查版本
            version = ultralytics.__version__
            print(f"✅ ultralytics版本: {version}")
            
            # 尝试加载YOLOv11模型测试
            try:
                model = YOLO('yolo11n.pt')
                print("✅ YOLOv11支持确认")
                return True
            except:
                print("⚠️ 当前版本不支持YOLOv11，升级中...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
                return True
                
        except ImportError:
            print("📦 安装最新版ultralytics...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "ultralytics"])
                print("✅ ultralytics安装成功")
                return True
            except subprocess.CalledProcessError:
                print("❌ ultralytics安装失败")
                return False
    
    def train_yolo11_model(self, model_size="yolo11s", epochs=100, imgsz=640):
        """训练YOLOv11模型"""
        
        if not self.install_ultralytics():
            return False
        
        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ ultralytics导入失败")
            return False
        
        print(f"🚀 开始训练YOLOv11{model_size[6:]}模型...")
        print(f"📊 训练参数: epochs={epochs}, imgsz={imgsz}")
        print(f"🔥 YOLOv11相比v8的改进:")
        print(f"   • 更好的特征提取backbone")
        print(f"   • 改进的FPN结构")
        print(f"   • 更精确的检测头")
        print(f"   • 更好的训练策略")
        
        # 创建YOLOv11模型
        model = YOLO(f'{model_size}.pt')
        
        # YOLOv11优化训练参数
        results = model.train(
            data=self.yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            patience=30,  # 增加耐心值
            batch=16,
            device=0,  # 使用GPU
            project='yolo11_scalebar_runs',
            name='scalebar_detection_v11',
            save_period=10,
            plots=True,
            val=True,
            
            # YOLOv11特有的优化参数
            optimizer='AdamW',  # 更好的优化器
            lr0=0.01,          # 初始学习率
            lrf=0.1,           # 最终学习率比例
            momentum=0.937,     # SGD动量
            weight_decay=0.0005, # 权重衰减
            warmup_epochs=3,    # 预热轮数
            warmup_momentum=0.8, # 预热动量
            warmup_bias_lr=0.1,  # 预热偏置学习率
            
            # 数据增强 - 针对比例尺优化
            hsv_h=0.015,       # 色调增强
            hsv_s=0.7,         # 饱和度增强
            hsv_v=0.4,         # 亮度增强
            degrees=0,         # 旋转角度 (比例尺通常水平，不旋转)
            translate=0.1,     # 平移
            scale=0.5,         # 缩放
            shear=0,           # 剪切 (比例尺不适合剪切)
            perspective=0,     # 透视变换
            flipud=0,          # 上下翻转
            fliplr=0.5,        # 左右翻转
            mosaic=1.0,        # 马赛克增强
            mixup=0.1,         # 混合增强
            
            # 损失函数权重
            box=7.5,           # 边界框损失权重
            cls=0.5,           # 分类损失权重
            dfl=1.5,           # DFL损失权重
            
            # 其他优化
            amp=True,          # 自动混合精度
            fraction=1.0,      # 使用全部数据
            profile=False,     # 不进行性能分析
            freeze=None,       # 不冻结层
            multi_scale=True,  # 多尺度训练
            overlap_mask=True, # 重叠掩码
            mask_ratio=4,      # 掩码比例
            dropout=0.0,       # Dropout
        )
        
        print("✅ YOLOv11模型训练完成!")
        
        # 评估模型
        metrics = model.val()
        print(f"📊 YOLOv11验证结果:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        print(f"   精确率: {metrics.box.mp:.3f}")
        print(f"   召回率: {metrics.box.mr:.3f}")
        
        return True
    
    def compare_with_yolo8(self):
        """与YOLOv8性能对比"""
        print("📊 YOLOv11 vs YOLOv8 理论对比:")
        print("="*50)
        print("🔥 YOLOv11优势:")
        print("   • mAP提升: +2-5%")
        print("   • 推理速度: +10-15%")
        print("   • 参数效率: 更少参数达到更好效果")
        print("   • 小目标检测: 显著改进")
        print("   • 边界框回归: 更精确")
        print("   • 训练稳定性: 更好的收敛")
        print()
        print("🎯 对比例尺检测的特殊优势:")
        print("   • 细长目标检测优化")
        print("   • 更好的特征融合")
        print("   • 改进的损失函数")

class YOLO11ScaleBarPredictor:
    """YOLOv11比例尺预测器"""
    
    def __init__(self, model_path="yolo11_scalebar_runs/scalebar_detection_v11/weights/best.pt"):
        self.model_path = model_path
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✅ 加载YOLOv11模型: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model = None
    
    def predict_image(self, image_path, conf=0.5, save_result=True):
        """使用YOLOv11预测单张图片"""
        
        if self.model is None:
            print("❌ 模型未加载")
            return None
        
        print(f"🔍 YOLOv11预测图片: {image_path}")
        
        # YOLOv11预测
        results = self.model(image_path, conf=conf, verbose=False)
        
        # 解析结果
        predictions = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # 智能估算比例尺线段的端点
                    # YOLOv11提供更准确的边界框，我们可以更精确地估算端点
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # 判断比例尺方向（水平或垂直）
                    if bbox_width > bbox_height:  # 水平比例尺
                        line_x1 = x1 + bbox_width * 0.1
                        line_y1 = center_y
                        line_x2 = x2 - bbox_width * 0.1
                        line_y2 = center_y
                    else:  # 垂直比例尺
                        line_x1 = center_x
                        line_y1 = y1 + bbox_height * 0.1
                        line_x2 = center_x
                        line_y2 = y2 - bbox_height * 0.1
                    
                    predictions.append({
                        'bbox': (x1, y1, x2, y2),
                        'line': (line_x1, line_y1, line_x2, line_y2),
                        'confidence': confidence,
                        'length': np.sqrt((line_x2 - line_x1)**2 + (line_y2 - line_y1)**2),
                        'orientation': 'horizontal' if bbox_width > bbox_height else 'vertical'
                    })
        
        print(f"📊 YOLOv11检测到 {len(predictions)} 个比例尺")
        
        # 可视化结果
        if save_result and predictions:
            self._visualize_predictions(image_path, predictions)
        
        return predictions
    
    def _visualize_predictions(self, image_path, predictions):
        """可视化YOLOv11预测结果"""
        
        image = cv2.imread(image_path)
        if image is None:
            return
        
        for i, pred in enumerate(predictions):
            # 绘制边界框
            x1, y1, x2, y2 = [int(x) for x in pred['bbox']]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制比例尺线段
            lx1, ly1, lx2, ly2 = [int(x) for x in pred['line']]
            cv2.line(image, (lx1, ly1), (lx2, ly2), (0, 0, 255), 4)
            cv2.circle(image, (lx1, ly1), 6, (255, 0, 0), -1)
            cv2.circle(image, (lx2, ly2), 6, (255, 0, 0), -1)
            
            # 添加详细标签
            conf_text = f"YOLOv11: {pred['confidence']:.3f}"
            orient_text = f"Dir: {pred['orientation']}"
            cv2.putText(image, conf_text, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(image, orient_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            print(f"   比例尺 {i+1}: 置信度={pred['confidence']:.3f}, 长度={pred['length']:.1f}px, 方向={pred['orientation']}")
        
        # 保存结果
        output_path = f"yolo11_prediction_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)
        print(f"💾 YOLOv11预测结果保存到: {output_path}")

def main():
    """主函数"""
    
    print("🚀 YOLOv11比例尺检测器 - 最新最强版本")
    print("="*60)
    
    print("🔥 YOLOv11的主要改进:")
    print("   • 更强的backbone网络")
    print("   • 改进的特征金字塔")
    print("   • 更精确的检测头")
    print("   • 优化的训练策略")
    print("   • 更好的小目标检测")
    print("="*60)
    
    print("选择操作:")
    print("1. 转换数据为YOLOv11格式")
    print("2. 训练YOLOv11模型")
    print("3. 评估YOLOv11模型")
    print("4. 预测单张图片")
    print("5. 批量预测")
    print("6. 完整YOLOv11流程 (推荐)")
    print("7. YOLOv11 vs YOLOv8对比")
    
    try:
        choice = input("请输入选择 (1-7): ").strip()
        
        if choice == '1':
            # 转换数据
            converter = YOLO11ScaleBarConverter('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
            sample_count = converter.convert_to_yolo_format()
            print(f"✅ 数据转换完成，共 {sample_count} 个样本")
            
        elif choice == '2':
            # 训练YOLOv11模型
            trainer = YOLO11ScaleBarTrainer()
            trainer.train_yolo11_model(model_size="yolo11s", epochs=100)
            
        elif choice == '3':
            # 评估模型
            from yolo_length_evaluator import YOLOLengthEvaluator
            evaluator = YOLOLengthEvaluator("yolo11_scalebar_runs/scalebar_detection_v11/weights/best.pt")
            evaluator.evaluate_dataset('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
            
        elif choice == '4':
            # 预测单张图片
            img_path = input("输入图片路径: ").strip()
            predictor = YOLO11ScaleBarPredictor()
            predictor.predict_image(img_path)
            
        elif choice == '5':
            # 批量预测
            img_dir = input("输入图片目录: ").strip()
            predictor = YOLO11ScaleBarPredictor()
            # 实现批量预测逻辑
            print("批量预测功能开发中...")
            
        elif choice == '6':
            # 完整YOLOv11流程
            print("🚀 执行完整YOLOv11训练流程...")
            
            # 步骤1: 转换数据
            print("\n📋 步骤1: 转换数据为YOLOv11格式")
            converter = YOLO11ScaleBarConverter('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
            sample_count = converter.convert_to_yolo_format()
            
            if sample_count < 50:
                print("⚠️ 样本数量较少，可能影响训练效果")
            
            # 步骤2: 训练YOLOv11模型
            print("\n📋 步骤2: 训练YOLOv11模型")
            trainer = YOLO11ScaleBarTrainer()
            success = trainer.train_yolo11_model(model_size="yolo11s", epochs=80)
            
            if success:
                # 步骤3: 长度误差评估
                print("\n📋 步骤3: 长度误差评估")
                try:
                    from yolo_length_evaluator import YOLOLengthEvaluator
                    evaluator = YOLOLengthEvaluator("yolo11_scalebar_runs/scalebar_detection_v11/weights/best.pt")
                    evaluator.evaluate_dataset('scale_bar_labels_fixed.csv', 'inputs/lizi/images')
                except ImportError:
                    print("请先运行长度评估器")
                
                print("\n🎉 YOLOv11比例尺检测器训练完成!")
                print("💡 预期YOLOv11比YOLOv8性能提升2-5%")
            
        elif choice == '7':
            # 性能对比
            trainer = YOLO11ScaleBarTrainer()
            trainer.compare_with_yolo8()
            
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n⏹️ 操作被中断")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
