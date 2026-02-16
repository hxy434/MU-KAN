#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
粒子分析演示脚本 - 集成UKAN模型
展示如何使用粒子分析功能
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from particle_analyzer import ParticleAnalyzer
from scale_detector import ScaleDetectorWithPreprocessing
from mock_ocr import process as ocr_process
from improved_ocr_guided_detector import ImprovedOCRGuidedDetector
import time
import glob
import sys
import torch
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize
from PIL import Image
import random
from datetime import datetime

# 添加seg-MOGA路径

import archs
from dataset import Dataset

ACCESS_TOKEN = "24.a6279612cc9567f9aca9316df9294369.2592000.1767946592.282335-119495206"

# 创建输出文件夹
def create_output_directory():
    """创建带时间戳的输出文件夹"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"particle_analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 结果将保存到: {output_dir}")
    return output_dir

# 全局输出目录
OUTPUT_DIR = create_output_directory()

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_ukan_model():
    """加载UKAN模型"""
    # 加载配置文件
    config_path = 'D:/python-learn/K/seg-MOGA/outputs/lizi_UKAN/config.yml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    
    # 创建模型
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        use_Moga=config['use_Moga']
    )
    
    # 加载权重
    ckpt = torch.load('D:/python-learn/K/seg-MOGA/outputs/lizi_UKAN/model.pth', map_location='cpu')
    try:
        model.load_state_dict(ckpt)
    except Exception as e:
        print("加载权重时出现异常，使用非严格模式:")
        print("Exception:", e)
        model.load_state_dict(ckpt, strict=False)
    
    model.eval()
    return model, config

def preprocess_image_for_ukan(image, config):
    """使用与训练时完全一致的预处理"""
    # 转换为RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 使用与验证时完全一致的预处理
    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),  # 与val.py中的transforms.Normalize()一致
    ])
    
    # 应用变换
    augmented = val_transform(image=img)
    img = augmented['image']
    
    # 转换为tensor
    img = img.astype('float32') / 255
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = torch.from_numpy(img).unsqueeze(0)  # 添加batch维度
    
    return img

def ukan_inference(image, model, config):
    """使用UKAN模型进行推理"""
    # 预处理
    input_tensor = preprocess_image_for_ukan(image, config)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
        prob_map = torch.sigmoid(output).cpu().numpy()[0, 0]  # 取第一个batch，第一个通道
    
    # 二值化（与val.py一致）
    pred_mask = (prob_map >= 0.5).astype(np.uint8)
    
    # 调整回原始图像尺寸
    original_h, original_w = image.shape[:2]
    pred_mask_resized = cv2.resize(pred_mask, (original_w, original_h))
    
    return prob_map, pred_mask_resized

def extract_scale_length_from_ocr(image_path):
    result = ocr_process(image_path)
    print("OCR原始结果：", result)
    words_result = result.get("words_result", [])
    for item in words_result:
        words = item["words"].replace(' ', '')
        print(f"处理OCR文本: '{words}'")
        for unit in ["nm", "μm", "um"]:
            if unit in words:
                num_str = words.replace(unit, "")
                try:
                    value = float(num_str)
                    print(f"找到比例尺: {value} {unit} (原始识别)")
                    return value, unit
                except:
                    continue
    print("未找到比例尺信息")
    return None, None

def detect_particles_with_yolo(image, model_path='D:/python-learn/K/seg-MOGA/runs/segment/kan/weights/best.pt'):
    """使用YOLO模型检测粒子"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        results = model(image, conf=0.3, iou=0.5)
        return results[0]  # 返回第一张图片的结果
    except Exception as e:
        print(f"YOLO检测失败: {e}")
        return None

def calculate_particle_features_from_mask(mask, pixel_to_real_ratio=None, unit="μm"):
    """从mask计算粒子特征"""
    from scipy import ndimage
    
    # 连通区域分析
    labeled_mask, num_features = ndimage.label(mask > 0)
    
    particle_features = []
    
    for i in range(1, num_features + 1):
        # 提取单个连通区域
        single_mask = (labeled_mask == i).astype(np.uint8)
        
        # 计算面积（像素数）
        area_pixels = np.sum(single_mask)
        
        # 计算等效直径
        diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        
        # 转换为物理单位
        if pixel_to_real_ratio:
            area_real = area_pixels * (pixel_to_real_ratio ** 2)
            diameter_real = diameter_pixels * pixel_to_real_ratio
        else:
            area_real = area_pixels
            diameter_real = diameter_pixels
        
        # 计算周长
        contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0
        
        # 计算圆度
        circularity = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0
        
        particle_features.append({
            'id': i,
            'area_pixels': area_pixels,
            'area_real': area_real,
            'diameter_pixels': diameter_pixels,
            'diameter_real': diameter_real,
            'perimeter': perimeter,
            'circularity': circularity
        })
    
    return particle_features, labeled_mask

def calculate_particle_features(masks, pixel_to_real_ratio=None, unit="μm"):
    """计算粒子特征"""
    particle_features = []
    
    for i, mask in enumerate(masks):
        # 计算面积（像素数）
        area_pixels = np.sum(mask)
        
        # 计算等效直径
        diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        
        # 转换为物理单位
        if pixel_to_real_ratio:
            area_real = area_pixels * (pixel_to_real_ratio ** 2)
            diameter_real = diameter_pixels * pixel_to_real_ratio
        else:
            area_real = area_pixels
            diameter_real = diameter_pixels
        
        # 计算周长
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0
        
        # 计算圆度
        circularity = (4 * np.pi * area_pixels) / (perimeter ** 2) if perimeter > 0 else 0
        
        particle_features.append({
            'id': i,
            'area_pixels': area_pixels,
            'area_real': area_real,
            'diameter_pixels': diameter_pixels,
            'diameter_real': diameter_real,
            'perimeter': perimeter,
            'circularity': circularity
        })
    
    return particle_features

def plot_particle_distributions(particle_features, pixel_to_real_ratio=None, unit="μm", save_path=None):
    """绘制粒子分布图"""
    if not particle_features:
        print("没有检测到粒子，无法绘制分布图")
        return
    
    # 提取数据
    areas = [p['area_real'] for p in particle_features]
    diameters = [p['diameter_real'] for p in particle_features]
    circularities = [p['circularity'] for p in particle_features]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Particle Distribution Analysis', fontsize=24, fontweight='bold')
    
    # 1. 面积分布直方图
    axes[0, 0].hist(areas, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel(f'Particle Area ({unit}$^2$)', fontsize=20)
    axes[0, 0].set_ylabel('Frequency', fontsize=20)
    axes[0, 0].set_title('Particle Area Distribution', fontsize=22)
    axes[0, 0].tick_params(axis='both', labelsize=18)
    
    # 2. 直径分布直方图
    axes[0, 1].hist(diameters, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel(f'Particle Diameter ({unit})', fontsize=20)
    axes[0, 1].set_ylabel('Frequency', fontsize=20)
    axes[0, 1].set_title('Particle Diameter Distribution', fontsize=22)
    axes[0, 1].tick_params(axis='both', labelsize=18)
    
    # 3. 圆度分布直方图
    axes[1, 0].hist(circularities, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 0].set_xlabel('Circularity', fontsize=20)
    axes[1, 0].set_ylabel('Frequency', fontsize=20)
    axes[1, 0].set_title('Particle Circularity Distribution', fontsize=22)
    axes[1, 0].tick_params(axis='both', labelsize=18)
    
    # 4. 面积vs直径散点图
    axes[1, 1].scatter(diameters, areas, alpha=0.6, color='purple')
    axes[1, 1].set_xlabel(f'Particle Diameter ({unit})', fontsize=20)
    axes[1, 1].set_ylabel(f'Particle Area ({unit}$^2$)', fontsize=20)
    axes[1, 1].set_title('Area vs Diameter Relationship', fontsize=22)
    axes[1, 1].tick_params(axis='both', labelsize=18)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存: {save_path}")
    plt.close()

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALE_DETECTOR_PATH = os.path.join(BASE_DIR, 'scale_detector_kan.pth')

def analyze_sem_image(image_path):
    print(f"\n📸 分析SEM图像: {image_path}")
    print("=" * 40)
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    print(f"图像尺寸: {image.shape}")
    
    # 1. 使用改进的OCR引导检测器检测比例尺
    coords, scale_length_pixels, confidence, scale_length_real, unit = detect_scale_with_improved_detector(image_path)
    
    # 显示详细的坐标信息
    if coords is not None:
        print(f"🔍 比例尺坐标信息:")
        print(f"   检测置信度: {confidence:.3f}")
        print(f"   比例尺起点坐标: ({coords[0]}, {coords[1]})")
        print(f"   比例尺终点坐标: ({coords[2]}, {coords[3]})")
        print(f"   比例尺像素长度: {scale_length_pixels:.1f} 像素")
        
        # 计算比例尺角度
        dx = coords[2] - coords[0]
        dy = coords[3] - coords[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"   比例尺角度: {angle:.1f}°")
    else:
        print("❌ 未检测到比例尺")
        # 如果改进检测器失败，尝试使用原有的检测器作为备选
        print("🔄 尝试使用原有检测器作为备选...")
        scale_detector = ScaleDetectorWithPreprocessing(
            model_path=SCALE_DETECTOR_PATH,
            input_size=(320, 320),
            hidden_dim=64
        )
        coords, scale_length_pixels, confidence = scale_detector.detect_scale(image)
        scale_length_real, unit = extract_scale_length_from_ocr(image_path)
    
    # 2. 自动换算
    if scale_length_pixels and scale_length_real:
        pixel_to_real_ratio = scale_length_real / scale_length_pixels
        print(f"每像素物理长度: {pixel_to_real_ratio} {unit}")
    else:
        pixel_to_real_ratio = None
        unit = "None"  # 没有检测到比例尺
        print("每像素长度: 未知（仅以像素为单位）")
    
    # 3. 初始化粒子分析器（用于备选方案）
    scale_detector = ScaleDetectorWithPreprocessing(
        model_path=SCALE_DETECTOR_PATH,
        input_size=(320, 320),
        hidden_dim=64
    )
    analyzer = ParticleAnalyzer(scale_detector)
    
    # 4. 使用UKAN模型检测粒子
    print("\n🔍 使用UKAN模型检测粒子...")
    try:
        # 加载UKAN模型
        ukan_model, ukan_config = load_ukan_model()
        
        # UKAN推理
        prob_map, pred_mask = ukan_inference(image, ukan_model, ukan_config)
        
        # 统计信息
        total_pixels = pred_mask.size
        positive_pixels = np.sum(pred_mask > 0)
        positive_ratio = positive_pixels / total_pixels * 100
        
        print(f"UKAN检测结果:")
        print(f"   总像素数: {total_pixels}")
        print(f"   正像素数: {positive_pixels}")
        print(f"   正像素比例: {positive_ratio:.4f}%")
        
        # 计算粒子特征
        particle_features, labeled_mask = calculate_particle_features_from_mask(
            pred_mask,
            pixel_to_real_ratio,
            unit if unit else "μm"
        )
        
        if particle_features:
            print(f"   检测到粒子数: {len(particle_features)}")
            
            # 6. 绘制分布图
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
            plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
            
            # 7. 输出统计信息
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            
            print(f"\n📊 UKAN粒子检测结果:")
            print(f"   检测到粒子数: {len(particle_features)}")
            print(f"   平均面积: {np.mean(areas):.2f} {unit}²")
            print(f"   平均直径: {np.mean(diameters):.2f} {unit}")
            print(f"   平均圆度: {np.mean(circularities):.3f}")
            print(f"   面积标准差: {np.std(areas):.2f} {unit}²")
            print(f"   直径标准差: {np.std(diameters):.2f} {unit}")
            
            # 保存可视化结果
            save_visualization_results(image, prob_map, pred_mask, base_name, particle_features, labeled_mask)
            
            # 保存分析报告
            save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "MU-KAN", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
            
        else:
            print("❌ UKAN未检测到粒子")
            # 保存分析报告（即使没有检测到粒子）
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "MU-KAN", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
            
    except Exception as e:
        print(f"❌ UKAN检测失败: {e}")
        print("尝试使用YOLO检测...")
        
        # 使用YOLO作为备选
        yolo_results = detect_particles_with_yolo(image)
        
        if yolo_results:
            print(f"YOLO检测到 {len(yolo_results.masks)} 个粒子")
            
            # 计算粒子特征
            particle_features = calculate_particle_features(
                yolo_results.masks.data.cpu().numpy(),
                pixel_to_real_ratio,
                unit if unit else "μm"
            )
            
            # 绘制分布图
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
            plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
            
            # 保存YOLO masks
            masks = yolo_results.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_mask_{i+1}.png")
                mask_uint8 = (mask * 255).astype(np.uint8)
                cv2.imwrite(mask_path, mask_uint8)
                print(f"💾 YOLO mask {i+1} 已保存: {mask_path}")
            
            # 保存合并的mask
            combined_mask = np.max(masks, axis=0).astype(np.uint8)
            combined_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_combined_mask.png")
            combined_mask_uint8 = (combined_mask * 255).astype(np.uint8)
            cv2.imwrite(combined_mask_path, combined_mask_uint8)
            print(f"💾 YOLO合并mask已保存: {combined_mask_path}")
            
            # 输出统计信息
            if particle_features:
                areas = [p['area_real'] for p in particle_features]
                diameters = [p['diameter_real'] for p in particle_features]
                circularities = [p['circularity'] for p in particle_features]
                
                print(f"\n📊 YOLO粒子检测结果:")
                print(f"   检测到粒子数: {len(particle_features)}")
                print(f"   平均面积: {np.mean(areas):.2f} {unit}²")
                print(f"   平均直径: {np.mean(diameters):.2f} {unit}")
                print(f"   平均圆度: {np.mean(circularities):.3f}")
                print(f"   面积标准差: {np.std(areas):.2f} {unit}²")
                print(f"   直径标准差: {np.std(diameters):.2f} {unit}")
                
                # 保存分析报告
                save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "YOLO", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
        else:
            print("❌ YOLO也未检测到粒子，使用传统方法...")
            # 使用原有的分析方法作为备选
            results = analyzer.analyze_image_with_scale(
                image,
                scale_length_real=scale_length_real,
                unit=unit if unit else "μm",
                particle_method="watershed",
                save_path=os.path.join(OUTPUT_DIR, f"sem_analysis_{os.path.basename(image_path)}.jpg")
            )
            # 输出分析结果
            if results:
                print(f"\n📊 传统方法分析结果:")
                print(f"   比例尺长度: {results['scale_info']['length_pixels']:.1f}像素")
                print(f"   检测到粒子数: {results['analysis']['count']}")
                print(f"   平均直径: {results['analysis']['diameter_stats']['mean']:.2f}{results['scale_info']['unit']}")
                print(f"   平均面积: {results['analysis']['area_stats']['mean']:.2f}{results['scale_info']['unit']}²")
                print(f"   平均圆度: {results['analysis']['circularity_stats']['mean']:.3f}")
                
                # 保存分析报告
                save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "传统方法", {"scale_length_pixels": scale_length_pixels, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
    
    print("\n✅ 分析完成！")

# 在analyze_sem_image_ukan_only和analyze_sem_image_yolo_only中，构造scale_info并返回

def analyze_sem_image_ukan_only(image_path):
    """仅使用UKAN模型进行分析"""
    print(f"\n📸 使用UKAN模型分析SEM图像: {image_path}")
    print("=" * 40)
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    scale_info = {}
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    print(f"图像尺寸: {image.shape}")
    
    # 1. 使用改进的OCR引导检测器检测比例尺
    coords, scale_length_pixels, confidence, scale_length_real, unit = detect_scale_with_improved_detector(image_path)
    
    # 显示详细的坐标信息
    if coords is not None:
        print(f"🔍 比例尺坐标信息:")
        print(f"   检测置信度: {confidence:.3f}")
        print(f"   比例尺起点坐标: ({coords[0]}, {coords[1]})")
        print(f"   比例尺终点坐标: ({coords[2]}, {coords[3]})")
        print(f"   比例尺像素长度: {scale_length_pixels:.1f} 像素")
        
        # 计算比例尺角度
        dx = coords[2] - coords[0]
        dy = coords[3] - coords[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"   比例尺角度: {angle:.1f}°")
    else:
        print("❌ 未检测到比例尺")
        # 如果改进检测器失败，尝试使用原有的检测器作为备选
        print("🔄 尝试使用原有检测器作为备选...")
        scale_detector = ScaleDetectorWithPreprocessing(
            model_path=SCALE_DETECTOR_PATH,
            input_size=(320, 320),
            hidden_dim=64
        )
        coords, scale_length_pixels, confidence = scale_detector.detect_scale(image)
        scale_length_real, unit = extract_scale_length_from_ocr(image_path)
    
    print(f"OCR原始识别结果: {scale_length_real} {unit}")
    
    # 2. 自动换算
    if scale_length_pixels and scale_length_real:
        pixel_to_real_ratio = scale_length_real / scale_length_pixels
        print(f"每像素物理长度: {pixel_to_real_ratio} {unit}")
        scale_info = {
            "scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None,
            "scale_length_real": scale_length_real,
            "unit": unit,
            "pixel_to_real_ratio": float(pixel_to_real_ratio) if pixel_to_real_ratio is not None else None,
            "coords": coords if coords is not None else None
        }
    else:
        pixel_to_real_ratio = None
        unit = "None"  # 没有检测到比例尺
        print("每像素长度: 未知（仅以像素为单位）")
        scale_info = {}
    # 5. 使用UKAN模型检测粒子
    print("\n🔍 使用UKAN模型检测粒子...")
    try:
        ukan_model, ukan_config = load_ukan_model()
        prob_map, pred_mask = ukan_inference(image, ukan_model, ukan_config)
        total_pixels = pred_mask.size
        positive_pixels = np.sum(pred_mask > 0)
        positive_ratio = positive_pixels / total_pixels * 100
        print(f"UKAN检测结果:")
        print(f"   总像素数: {total_pixels}")
        print(f"   正像素数: {positive_pixels}")
        print(f"   正像素比例: {positive_ratio:.4f}%")
        particle_features, labeled_mask = calculate_particle_features_from_mask(
            pred_mask,
            pixel_to_real_ratio,
            unit if unit else "μm"
        )
        if particle_features:
            print(f"   检测到粒子数: {len(particle_features)}")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
            plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            print(f"\n📊 UKAN粒子检测结果:")
            print(f"   检测到粒子数: {len(particle_features)}")
            print(f"   平均面积: {np.mean(areas):.2f} {unit}²")
            print(f"   平均直径: {np.mean(diameters):.2f} {unit}")
            print(f"   平均圆度: {np.mean(circularities):.3f}")
            print(f"   面积标准差: {np.std(areas):.2f} {unit}²")
            print(f"   直径标准差: {np.std(diameters):.2f} {unit}")
            save_visualization_results(image, prob_map, pred_mask, base_name, particle_features, labeled_mask)
            save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "MU-KAN Segmentation and YOLOv11 Detection", scale_info)
        else:
            print("❌ UKAN未检测到粒子")
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "MU-KAN", scale_info)
        print("\n✅ 分析完成！")
        print(f"分析结果目录: {OUTPUT_DIR}")
        return OUTPUT_DIR, {"scale_info": scale_info}
    except Exception as e:
        print(f"❌ UKAN检测失败: {e}")
        print("UKAN检测失败，请检查模型文件或尝试其他方法")
        print("\n✅ 分析完成！")
        print(f"分析结果目录: {OUTPUT_DIR}")
        return OUTPUT_DIR, {"scale_info": scale_info}

def analyze_sem_image_yolo_only(image_path):
    """仅使用YOLO模型进行分析"""
    print(f"\n📸 使用YOLO模型分析SEM图像: {image_path}")
    print("=" * 40)
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    print(f"图像尺寸: {image.shape}")
    
    # 1. 使用改进的OCR引导检测器检测比例尺
    coords, scale_length_pixels, confidence, scale_length_real, unit = detect_scale_with_improved_detector(image_path)
    
    # 显示详细的坐标信息
    if coords is not None:
        print(f"🔍 比例尺坐标信息:")
        print(f"   检测置信度: {confidence:.3f}")
        print(f"   比例尺起点坐标: ({coords[0]}, {coords[1]})")
        print(f"   比例尺终点坐标: ({coords[2]}, {coords[3]})")
        print(f"   比例尺像素长度: {scale_length_pixels:.1f} 像素")
        
        # 计算比例尺角度
        dx = coords[2] - coords[0]
        dy = coords[3] - coords[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"   比例尺角度: {angle:.1f}°")
    else:
        print("❌ 未检测到比例尺")
        # 如果改进检测器失败，尝试使用原有的检测器作为备选
        print("🔄 尝试使用原有检测器作为备选...")
        scale_detector = ScaleDetectorWithPreprocessing(
            model_path=SCALE_DETECTOR_PATH,
            input_size=(320, 320),
            hidden_dim=64
        )
        coords, scale_length_pixels, confidence = scale_detector.detect_scale(image)
        scale_length_real, unit = extract_scale_length_from_ocr(image_path)
    
    print(f"OCR原始识别结果: {scale_length_real} {unit}")
    
    # 2. 自动换算
    if scale_length_pixels and scale_length_real:
        pixel_to_real_ratio = scale_length_real / scale_length_pixels
        print(f"每像素物理长度: {pixel_to_real_ratio} {unit}")
    else:
        pixel_to_real_ratio = None
        unit = "None"  # 没有检测到比例尺
        print("每像素长度: 未知（仅以像素为单位）")
    
    # 5. 使用YOLO检测粒子
    print("\n🔍 使用YOLO检测粒子...")
    yolo_results = detect_particles_with_yolo(image)
    
    if yolo_results:
        print(f"YOLO检测到 {len(yolo_results.masks)} 个粒子")
        
        # 计算粒子特征
        particle_features = calculate_particle_features(
            yolo_results.masks.data.cpu().numpy(),
            pixel_to_real_ratio,
            unit if unit else "μm"
        )
        
        # 绘制分布图
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        distribution_path = os.path.join(OUTPUT_DIR, f"particle_distribution_{base_name}.png")
        plot_particle_distributions(particle_features, pixel_to_real_ratio, unit, distribution_path)
        
        # 保存YOLO masks
        masks = yolo_results.masks.data.cpu().numpy()
        for i, mask in enumerate(masks):
            mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_mask_{i+1}.png")
            mask_uint8 = (mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, mask_uint8)
            print(f"💾 YOLO mask {i+1} 已保存: {mask_path}")
        
        # 保存合并的mask
        combined_mask = np.max(masks, axis=0).astype(np.uint8)
        combined_mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_yolo_combined_mask.png")
        combined_mask_uint8 = (combined_mask * 255).astype(np.uint8)
        cv2.imwrite(combined_mask_path, combined_mask_uint8)
        print(f"💾 YOLO合并mask已保存: {combined_mask_path}")
        
        # 输出统计信息
        if particle_features:
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            
            print(f"\n📊 YOLO粒子检测结果:")
            print(f"   检测到粒子数: {len(particle_features)}")
            print(f"   平均面积: {np.mean(areas):.2f} {unit}²")
            print(f"   平均直径: {np.mean(diameters):.2f} {unit}")
            print(f"   平均圆度: {np.mean(circularities):.3f}")
            print(f"   面积标准差: {np.std(areas):.2f} {unit}²")
            print(f"   直径标准差: {np.std(diameters):.2f} {unit}")
            
            # 保存分析报告
            save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, "YOLO", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
        else:
            print("❌ YOLO未检测到粒子")
            # 保存分析报告（即使没有检测到粒子）
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "YOLO", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
    else:
        print("❌ YOLO检测失败，使用传统方法...")
        # 使用原有的分析方法作为备选
        results = analyzer.analyze_image_with_scale(
            image,
            scale_length_real=scale_length_real,
            unit=unit if unit else "μm",
            particle_method="watershed",
            save_path=os.path.join(OUTPUT_DIR, f"sem_analysis_{os.path.basename(image_path)}.jpg")
        )
        # 输出分析结果
        if results:
            print(f"\n📊 传统方法分析结果:")
            print(f"   比例尺长度: {results['scale_info']['length_pixels']:.1f}像素")
            print(f"   检测到粒子数: {results['analysis']['count']}")
            print(f"   平均直径: {results['analysis']['diameter_stats']['mean']:.2f}{results['scale_info']['unit']}")
            print(f"   平均面积: {results['analysis']['area_stats']['mean']:.2f}{results['scale_info']['unit']}²")
            print(f"   平均圆度: {results['analysis']['circularity_stats']['mean']:.3f}")
                
            # 保存分析报告
            save_analysis_report(image_path, [], pixel_to_real_ratio, unit, "传统方法", {"scale_length_pixels": float(scale_length_pixels) if scale_length_pixels is not None else None, "scale_length_real": scale_length_real, "unit": unit, "coords": coords})
    
    print("\n✅ 分析完成！")
    print(f"分析结果目录: {OUTPUT_DIR}")
    return OUTPUT_DIR

def detect_scale_with_improved_detector(image_path, access_token=ACCESS_TOKEN):
    """
    使用改进的OCR引导检测器检测比例尺
    返回: (coords, scale_length_pixels, confidence, scale_length_real, unit)
    """
    print(f"🔍 使用改进的OCR引导检测器检测比例尺...")
    
    # 初始化改进的检测器
    model_path = "../epoch80.pt"  # 权重文件在根目录下，从backend目录运行时需要回到上级目录
    try:
        detector = ImprovedOCRGuidedDetector(model_path, access_token)
    except Exception as e:
        print(f"❌ 改进检测器初始化失败: {e}")
        return None, None, None, None, None
    
    # 进行混合检测
    try:
        detection_result = detector.detect_scale_hybrid(image_path, conf=0.2)
        
        if not detection_result['final_results']:
            print("❌ 改进检测器未检测到比例尺")
            return None, None, None, None, None
        
        # 获取最佳检测结果
        best_result = detection_result['final_results'][0]
        bbox = best_result['bbox']
        
        # 计算比例尺坐标和长度
        x1, y1, x2, y2 = bbox
        coords = [int(x1), int(y1), int(x2), int(y2)]
        scale_length_pixels = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        confidence = float(best_result.get('confidence', 0.5))
        
        # 从OCR结果中提取物理长度和单位
        scale_length_real = None
        unit = None
        
        # 尝试从文本区域中提取比例尺信息
        for region in detection_result['text_regions']:
            text = region['text']
            # 使用正则表达式提取数字和单位
            import re
            pattern = r'(\d+(?:\.\d+)?)\s*(nm|μm|um|mm|cm|m)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                scale_length_real = float(match.group(1))
                unit = match.group(2)
                break
        
        # 如果OCR没有提取到，尝试使用原有的OCR函数
        if scale_length_real is None:
            scale_length_real, unit = extract_scale_length_from_ocr(image_path)
        
        print(f"✅ 改进检测器检测成功:")
        print(f"   比例尺坐标: ({coords[0]}, {coords[1]}) -> ({coords[2]}, {coords[3]})")
        print(f"   像素长度: {scale_length_pixels:.1f}")
        print(f"   置信度: {confidence:.3f}")
        print(f"   物理长度: {scale_length_real} {unit}")
        
        # 保存可视化结果
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_path = os.path.join(OUTPUT_DIR, f"{base_name}_improved_detection.png")
        detector.visualize_results(detection_result, vis_path)
        
        return coords, scale_length_pixels, confidence, scale_length_real, unit
        
    except Exception as e:
        print(f"❌ 改进检测器检测失败: {e}")
        return None, None, None, None, None

def save_visualization_results(image, prob_map, pred_mask, base_name, particle_features=None, labeled_mask=None):
    """保存可视化结果，并在原图上标注粒子编号和分割边界"""
    # 使用全局输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存二值mask
    mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_binary_mask.png")
    mask_uint8 = (pred_mask * 255).astype(np.uint8)
    cv2.imwrite(mask_path, mask_uint8)
    print(f"💾 二值mask已保存: {mask_path}")
    
    # 保存概率图
    prob_path = os.path.join(OUTPUT_DIR, f"{base_name}_probability_map.png")
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(prob_path, prob_uint8)
    print(f"💾 概率图已保存: {prob_path}")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'MU-KAN Model Analysis Results - {base_name}', fontsize=16, fontweight='bold')
    
    # 1. 原始图像
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 2. 概率图
    axes[0, 1].imshow(prob_map, cmap='hot')
    axes[0, 1].set_title('Probability Map')
    axes[0, 1].axis('off')
    
    # 3. 二值化mask
    axes[1, 0].imshow(pred_mask, cmap='gray')
    axes[1, 0].set_title('Binary Mask (Threshold 0.5)')
    axes[1, 0].axis('off')
    
    # 4. 叠加显示
    overlay = original_rgb.copy()
    mask_resized = cv2.resize(pred_mask, (original_rgb.shape[1], original_rgb.shape[0]))
    overlay[mask_resized > 0] = [0, 255, 0]  # 绿色显示检测区域
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay Display (Green=Detection Area)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, f"{base_name}_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 可视化结果已保存: {save_path}")
    plt.close()

    # 保存带编号的原图
    if particle_features is not None and labeled_mask is not None and len(particle_features) > 0:
        numbered_img = image.copy()
        for idx, feat in enumerate(particle_features, 1):
            mask = (labeled_mask == idx).astype(np.uint8)
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.putText(numbered_img, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)
        numbered_path = os.path.join(OUTPUT_DIR, f"{base_name}_numbered.png")
        cv2.imwrite(numbered_path, numbered_img)
        print(f"💾 带编号原图已保存: {numbered_path}")
    # 保存原图+分割边界叠加图
    contour_img = image.copy()
    contours, _ = cv2.findContours((pred_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)  # 红色线条
    contour_path = os.path.join(OUTPUT_DIR, f"{base_name}_contour.png")
    cv2.imwrite(contour_path, contour_img)
    print(f"💾 分割边界叠加图已保存: {contour_path}")

def save_analysis_report(image_path, particle_features, pixel_to_real_ratio, unit, method="MU-KAN", scale_info=None):
    report_path = os.path.join(OUTPUT_DIR, f"analysis_report_{os.path.splitext(os.path.basename(image_path))[0]}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Particle Analysis Report\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Method: {method}\n")
        f.write(f"Scale Unit: {unit}\n")
        if scale_info:
            f.write(f"Scale Length (pixels): {scale_info.get('scale_length_pixels', 'Not detected')}\n")
            f.write(f"OCR Detected Physical Length: {scale_info.get('scale_length_real', 'Not detected')} {scale_info.get('unit', '')}\n")
            # 添加坐标信息到报告
            if 'coords' in scale_info:
                coords = scale_info['coords']
                f.write(f"Scale Start Coordinates: ({coords[0]}, {coords[1]})\n")
                f.write(f"Scale End Coordinates: ({coords[2]}, {coords[3]})\n")
                dx = coords[2] - coords[0]
                dy = coords[3] - coords[1]
                angle = np.arctan2(dy, dx) * 180 / np.pi
                f.write(f"Scale Angle: {angle:.1f}°\n")
        f.write(f"Pixel to Physical Unit Ratio: {pixel_to_real_ratio}\n\n")
        if particle_features:
            areas = [p['area_real'] for p in particle_features]
            diameters = [p['diameter_real'] for p in particle_features]
            circularities = [p['circularity'] for p in particle_features]
            f.write(f"Detection Results Statistics:\n")
            f.write(f"  Number of Particles Detected: {len(particle_features)}\n")
            f.write(f"  Average Area: {np.mean(areas):.2f} {unit}²\n")
            f.write(f"  Average Diameter: {np.mean(diameters):.2f} {unit}\n")
            f.write(f"  Average Circularity: {np.mean(circularities):.3f}\n")
            f.write(f"  Area Standard Deviation: {np.std(areas):.2f} {unit}²\n")
            f.write(f"  Diameter Standard Deviation: {np.std(diameters):.2f} {unit}\n\n")
            f.write(f"Detailed Particle Information:\n")
            for i, particle in enumerate(particle_features):
                f.write(f"  Particle {i+1}:\n")
                f.write(f"    Area: {particle['area_real']:.2f} {unit}²\n")
                f.write(f"    Diameter: {particle['diameter_real']:.2f} {unit}\n")
                f.write(f"    Circularity: {particle['circularity']:.3f}\n")
        else:
            f.write("No particles detected\n")
    print(f"📄 Analysis report saved: {report_path}")

if __name__ == "__main__":
    print("🎯 全自动KAN比例尺+粒子检测+分布图分析 (批量模式)")
    print("=" * 50)
    
    # 选择检测方法
    print("请选择粒子检测方法：")
    print("1 = UKAN语义分割 (推荐，效果更好)")
    print("2 = YOLO目标检测 (备选方案)")
    print("3 = 自动选择 (优先UKAN，失败时使用YOLO)")
    
    mode = input("请输入选择 (1/2/3): ").strip()
    
    # 支持批量处理文件夹下所有图片
    folder = input("请输入SEM图像文件夹路径（如 datasets/lizi/images 或单张图片路径）: ").strip()
    
    # 去除可能的引号
    folder = folder.strip('"\'')
    
    if os.path.isdir(folder):
        # 处理所有png/jpg图片
        image_list = sorted(glob.glob(folder + "/*.png") + glob.glob(folder + "/*.jpg"))
        print(f"共检测到 {len(image_list)} 张图片，将依次分析...")
        for image_path in image_list:
            if mode == "1":
                analyze_sem_image_ukan_only(image_path)
            elif mode == "2":
                analyze_sem_image_yolo_only(image_path)
            elif mode == "3":
                analyze_sem_image(image_path)
            else:
                print("无效选择，使用自动模式")
                analyze_sem_image(image_path)
            time.sleep(0.7)  # 防止百度OCR QPS超限
    else:
        # 单张图片
        if mode == "1":
            analyze_sem_image_ukan_only(folder)
        elif mode == "2":
            analyze_sem_image_yolo_only(folder)
        elif mode == "3":
            analyze_sem_image(folder)
        else:
            print("无效选择，使用自动模式")
            analyze_sem_image(folder) 
    
