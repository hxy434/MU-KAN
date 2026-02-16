#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT) 比例尺检测模型
专门训练ViT模型，200轮训练，不早停
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import json

# 导入你的百度OCR模块
try:
    import mock_ocr
    OCR_AVAILABLE = True
    print("✅ 百度OCR模块已加载")
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ 百度OCR模块未找到，将使用全图检测")

class ViTScaleBarDataset(Dataset):
    """ViT比例尺数据集 - 集成OCR辅助检测"""
    
    def __init__(self, csv_file, img_dir, transform=None, split='train', use_ocr=True):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        self.use_ocr = use_ocr and OCR_AVAILABLE
        
        # 计算比例尺长度
        self.data['scale_length'] = abs(self.data['x2'] - self.data['x1'])
        
        # 过滤异常值
        self.data = self.data[self.data['scale_length'] > 0]
        self.data = self.data[self.data['scale_length'] >= 5]
        self.data = self.data[self.data['scale_length'] <= 1000]
        
        print(f"📊 数据过滤后: {len(self.data)} 个样本")
        
        # 重置索引
        self.data = self.data.reset_index(drop=True)
        
        # 分层采样划分数据集
        self._create_split()
        
        # 计算长度统计
        self.length_mean = self.data['scale_length'].mean()
        self.length_std = self.data['scale_length'].std()
        print(f"   长度统计: 均值={self.length_mean:.2f}, 标准差={self.length_std:.2f}")
        
        # OCR辅助信息存储
        self.ocr_info = {}
        if self.use_ocr:
            print("🔤 启用OCR辅助检测...")
            self._extract_ocr_info()
        else:
            print("📏 使用全图检测模式")
        
    def _create_split(self):
        """分层采样创建训练/验证集"""
        # 按长度分组
        short_scales = self.data[self.data['scale_length'] < 100]
        medium_scales = self.data[(self.data['scale_length'] >= 100) & (self.data['scale_length'] < 200)]
        long_scales = self.data[self.data['scale_length'] >= 200]
        
        train_indices = []
        val_indices = []
        
        # 从每组中按8:2比例分配
        for scale_group in [short_scales, medium_scales, long_scales]:
            if len(scale_group) > 0:
                np.random.seed(42)
                indices = scale_group.index.tolist()
                np.random.shuffle(indices)
                
                train_n = int(len(indices) * 0.8)
                train_indices.extend(indices[:train_n])
                val_indices.extend(indices[train_n:])
        
        # 根据split选择对应的索引
        if self.split == 'train':
            self.indices = train_indices
        else:
            self.indices = val_indices
    
    def _extract_ocr_info(self):
        """使用OCR提取比例尺信息"""
        print("🔍 使用百度OCR提取比例尺信息...")
        
        ocr_success_count = 0
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data), desc="OCR处理"):
            img_path = os.path.join(self.img_dir, str(row['filename']))
            
            if not os.path.exists(img_path):
                continue
            
            try:
                # 使用你的百度OCR
                scale_length, unit = mock_ocr.extract_scale_length_from_ocr(img_path)
                
                if scale_length is not None and unit is not None:
                    # 确保scale_length是字符串类型
                    scale_length_str = str(scale_length)
                    unit_str = str(unit)
                    
                    # 获取OCR文字位置
                    ocr_result = mock_ocr.process(img_path)
                    text_region = self._find_scale_text_region(ocr_result, scale_length_str, unit_str)
                    
                    self.ocr_info[row['filename']] = {
                        'scale_length': scale_length_str,
                        'unit': unit_str,
                        'text_region': text_region,
                        'ocr_success': True
                    }
                    ocr_success_count += 1
                else:
                    self.ocr_info[row['filename']] = {
                        'ocr_success': False
                    }
                    
            except Exception as e:
                print(f"⚠️ OCR处理失败 {row['filename']}: {e}")
                self.ocr_info[row['filename']] = {
                    'ocr_success': False
                }
        
        print(f"✅ OCR处理完成: {ocr_success_count}/{len(self.data)} 成功 ({ocr_success_count/len(self.data)*100:.1f}%)")
    
    def _find_scale_text_region(self, ocr_result, scale_length, unit):
        """在OCR结果中查找比例尺文字区域"""
        if ocr_result.get("error_code", 0) != 0:
            return None
        
        words_result = ocr_result.get("words_result", [])
        target_text = f"{scale_length}{unit}"
        
        for item in words_result:
            words = item.get("words", "")
            location = item.get("location", {})
            
            # 确保words是字符串类型
            if not isinstance(words, str):
                continue
            
            # 检查是否包含目标文字
            if target_text in words or str(scale_length) in words:
                return {
                    'left': location.get("left", 0),
                    'top': location.get("top", 0),
                    'width': location.get("width", 0),
                    'height': location.get("height", 0)
                }
        
        return None
    
    def _expand_search_region(self, text_region, image_shape):
        """基于OCR文字区域扩展搜索区域"""
        if text_region is None:
            return None
        
        h, w = image_shape[:2]
        
        # 获取文字区域
        left = text_region['left']
        top = text_region['top']
        width = text_region['width']
        height = text_region['height']
        
        # 扩展搜索区域 - 向左右扩展
        expand_factor = 3.0  # 扩展3倍
        expand_width = int(width * expand_factor)
        expand_height = int(height * 2)  # 高度扩展2倍
        
        # 计算扩展后的坐标
        center_x = left + width // 2
        center_y = top + height // 2
        
        new_left = max(0, center_x - expand_width // 2)
        new_right = min(w, center_x + expand_width // 2)
        new_top = max(0, center_y - expand_height // 2)
        new_bottom = min(h, center_y + expand_height // 2)
        
        return (new_left, new_top, new_right, new_bottom)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取数据
        row_idx = self.indices[idx]
        row = self.data.iloc[row_idx]
        filename = str(row['filename'])
        
        # 读取图片
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # OCR辅助裁剪
        if self.use_ocr and filename in self.ocr_info:
            ocr_info = self.ocr_info[filename]
            
            if ocr_info.get('ocr_success', False) and ocr_info.get('text_region'):
                # 基于OCR文字区域扩展搜索
                search_region = self._expand_search_region(ocr_info['text_region'], image.shape)
                
                if search_region is not None:
                    left, top, right, bottom = search_region
                    # 裁剪搜索区域
                    cropped_image = image[top:bottom, left:right]
                    
                    if cropped_image.size > 0:  # 确保裁剪有效
                        image = cropped_image
                        # 只在训练时显示裁剪信息，避免过多输出
                        if self.split == 'train' and idx % 50 == 0:  # 每50个样本显示一次
                            print(f"🔍 {filename}: 使用OCR辅助裁剪 {image.shape}")
        
        # 数据增强
        if self.transform:
            image = self.transform(image)
        
        # 获取原始长度
        original_length = abs(row['x2'] - row['x1'])
        
        # 使用对数变换标准化长度
        log_length = np.log1p(original_length)
        
        # 添加OCR信息
        ocr_assisted = False
        ocr_scale_info = {
            'scale_length': '',
            'unit': ''
        }
        
        if self.use_ocr and filename in self.ocr_info:
            ocr_info = self.ocr_info[filename]
            ocr_assisted = ocr_info.get('ocr_success', False)
            if ocr_assisted:
                ocr_scale_info = {
                    'scale_length': ocr_info.get('scale_length', ''),
                    'unit': ocr_info.get('unit', '')
                }
        
        return {
            'image': image,
            'original_length': torch.tensor(original_length, dtype=torch.float32),
            'log_length': torch.tensor(log_length, dtype=torch.float32),
            'filename': filename,
            'ocr_assisted': ocr_assisted,
            'ocr_scale_info': ocr_scale_info
        }

class ViTScaleDetector(nn.Module):
    """基于Vision Transformer的比例尺检测模型"""
    
    def __init__(self, model_name='vit_b_16', pretrained=True):
        super(ViTScaleDetector, self).__init__()
        
        # 使用torchvision的Vision Transformer
        if model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 768
        elif model_name == 'vit_l_16':
            self.backbone = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")
        
        # 移除最后的分类头
        self.backbone.heads = nn.Identity()
        
        # 冻结部分层，但保留更多层可训练
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # 优化的检测头 - 专门为ViT设计
        self.scale_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # 预测对数长度
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.scale_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        log_length = self.scale_head(features)
        return log_length.squeeze()

class ViTScaleLengthLoss(nn.Module):
    """ViT比例尺长度误差损失函数"""
    
    def __init__(self, alpha=0.7, beta=0.3):
        super(ViTScaleLengthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def forward(self, pred_log_lengths, gt_log_lengths, gt_original_lengths):
        # 对数空间的Smooth L1损失
        log_loss = self.smooth_l1_loss(pred_log_lengths, gt_log_lengths)
        
        # 转换回原始空间计算相对误差
        pred_original_lengths = torch.expm1(pred_log_lengths)
        relative_error = torch.mean(torch.abs(pred_original_lengths - gt_original_lengths) / gt_original_lengths)
        
        # 总损失
        total_loss = self.alpha * log_loss + self.beta * relative_error
        
        return total_loss, log_loss, relative_error

class ViTTrainer:
    """ViT比例尺检测训练器"""
    
    def __init__(self, model_name='vit_b_16'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.best_length_error = float('inf')
        self.best_model_path = None
        
        # ViT专用数据变换
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_datasets(self, csv_file, img_dir, use_ocr=True):
        """创建数据集"""
        print(f"📊 创建 ViT 数据集...")
        
        train_dataset = ViTScaleBarDataset(csv_file, img_dir, self.train_transform, 'train', use_ocr)
        val_dataset = ViTScaleBarDataset(csv_file, img_dir, self.val_transform, 'val', use_ocr)
        
        print(f"   训练集: {len(train_dataset)} 个样本")
        print(f"   验证集: {len(val_dataset)} 个样本")
        
        # 统计OCR辅助情况
        if use_ocr and OCR_AVAILABLE:
            train_ocr_count = sum(1 for i in range(len(train_dataset)) 
                                if train_dataset[i]['ocr_assisted'])
            val_ocr_count = sum(1 for i in range(len(val_dataset)) 
                              if val_dataset[i]['ocr_assisted'])
            
            print(f"   OCR辅助训练样本: {train_ocr_count}/{len(train_dataset)} ({train_ocr_count/len(train_dataset)*100:.1f}%)")
            print(f"   OCR辅助验证样本: {val_ocr_count}/{len(val_dataset)} ({val_ocr_count/len(val_dataset)*100:.1f}%)")
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset, val_dataset, epochs=200, batch_size=16, lr=1e-4):
        """训练模型 - 200轮，不早停，集成OCR辅助"""
        print(f"🚀 开始训练 {self.model_name} 模型...")
        print(f"   设备: {self.device}")
        print(f"   Epochs: {epochs} (不早停)")
        print(f"   Batch Size: {batch_size}")
        print(f"   学习率: {lr}")
        print(f"   OCR辅助: {'启用' if OCR_AVAILABLE else '未启用'}")
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # 创建模型
        model = ViTScaleDetector(model_name=self.model_name, pretrained=True)
        model.to(self.device)
        
        # 损失函数和优化器
        criterion = ViTScaleLengthLoss(alpha=0.7, beta=0.3)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        
        # 学习率调度器 - 不早停，但可以调整学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=15, min_lr=1e-6
        )
        
        # 训练历史
        train_losses = []
        val_losses = []
        length_errors = []
        
        # 创建保存目录
        os.makedirs('vit_models', exist_ok=True)
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            ocr_assisted_count = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch in train_pbar:
                images = batch['image'].to(self.device)
                gt_log_lengths = batch['log_length'].to(self.device)
                gt_original_lengths = batch['original_length'].to(self.device)
                
                # 统计OCR辅助样本
                if 'ocr_assisted' in batch:
                    ocr_assisted_count += sum(1 for assisted in batch['ocr_assisted'] if assisted)
                
                optimizer.zero_grad()
                
                pred_log_lengths = model(images)
                loss, log_loss, relative_error = criterion(pred_log_lengths, gt_log_lengths, gt_original_lengths)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'RelError': f'{relative_error.item()*100:.2f}%',
                    'OCR': f'{ocr_assisted_count}' if OCR_AVAILABLE else 'N/A'
                })
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            total_length_error = 0.0
            num_samples = 0
            val_ocr_assisted_count = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for batch in val_pbar:
                    images = batch['image'].to(self.device)
                    gt_log_lengths = batch['log_length'].to(self.device)
                    gt_original_lengths = batch['original_length'].to(self.device)
                    
                    # 统计OCR辅助样本
                    if 'ocr_assisted' in batch:
                        val_ocr_assisted_count += sum(1 for assisted in batch['ocr_assisted'] if assisted)
                    
                    pred_log_lengths = model(images)
                    loss, log_loss, relative_error = criterion(pred_log_lengths, gt_log_lengths, gt_original_lengths)
                    
                    val_loss += loss.item()
                    total_length_error += relative_error.item() * images.size(0)
                    num_samples += images.size(0)
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'RelError': f'{relative_error.item()*100:.2f}%',
                        'OCR': f'{val_ocr_assisted_count}' if OCR_AVAILABLE else 'N/A'
                    })
            
            val_loss /= len(val_loader)
            avg_length_error = total_length_error / num_samples * 100
            val_losses.append(val_loss)
            length_errors.append(avg_length_error)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 保存最佳模型（基于长度误差）
            if avg_length_error < self.best_length_error:
                self.best_length_error = avg_length_error
                self.best_model_path = f'vit_models/best_{self.model_name}_scale_detector.pth'
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_length_error': self.best_length_error,
                    'model_name': self.model_name
                }, self.best_model_path)
                
                print(f"💾 新的最佳模型! Epoch {epoch+1}, 长度误差: {self.best_length_error:.2f}%")
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'vit_models/checkpoint_{self.model_name}_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'length_error': avg_length_error,
                    'model_name': self.model_name
                }, checkpoint_path)
                print(f"💾 检查点保存: {checkpoint_path}")
            
            # 打印进度
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  长度误差: {avg_length_error:.2f}%")
            print(f"  最佳长度误差: {self.best_length_error:.2f}%")
            print(f"  学习率: {optimizer.param_groups[0]['lr']:.2e}")
            if OCR_AVAILABLE:
                print(f"  OCR辅助训练: {ocr_assisted_count} 样本")
                print(f"  OCR辅助验证: {val_ocr_assisted_count} 样本")
            print("-" * 50)
        
        # 绘制训练曲线
        self._plot_training_curves(train_losses, val_losses, length_errors)
        
        print(f"✅ 训练完成! 最佳模型保存在: {self.best_model_path}")
        print(f"🎯 最佳长度误差: {self.best_length_error:.2f}%")
        
        return model
    
    def _plot_training_curves(self, train_losses, val_losses, length_errors):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_title(f'{self.model_name.upper()} Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 长度误差曲线
        ax2.plot(length_errors, label='Length Error', color='green')
        ax2.set_title(f'{self.model_name.upper()} Scale Length Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'vit_models/{self.model_name}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model_on_dataset(self, model, csv_file, img_dir, use_ocr=True):
        """在完整数据集上评估模型并保存结果"""
        print(f"🔍 在完整数据集上评估模型...")
        
        # 创建完整数据集（不分割，使用所有数据）
        class FullDataset(ViTScaleBarDataset):
            def _create_split(self):
                """使用所有数据，不分割"""
                self.indices = list(range(len(self.data)))
        
        full_dataset = FullDataset(csv_file, img_dir, self.val_transform, 'val', use_ocr)
        full_loader = DataLoader(full_dataset, batch_size=16, shuffle=False, num_workers=4)
        
        # 加载最佳模型
        if self.best_model_path and os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 加载最佳模型: {self.best_model_path}")
        else:
            print("⚠️ 未找到最佳模型，使用当前模型")
        
        model.eval()
        
        # 存储结果
        results = []
        total_error = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(full_loader, desc="评估进度"):
                images = batch['image'].to(self.device)
                gt_original_lengths = batch['original_length'].to(self.device)
                filenames = batch['filename']
                ocr_assisted = batch.get('ocr_assisted', [False] * len(filenames))
                ocr_scale_info = batch.get('ocr_scale_info', [{'scale_length': '', 'unit': ''}] * len(filenames))
                
                # 预测
                pred_log_lengths = model(images)
                pred_original_lengths = torch.expm1(pred_log_lengths)
                
                # 计算每个样本的误差
                for i in range(len(filenames)):
                    pred_length = pred_original_lengths[i].item()
                    true_length = gt_original_lengths[i].item()
                    
                    # 计算误差
                    absolute_error = abs(pred_length - true_length)
                    relative_error = (absolute_error / true_length) * 100
                    
                    # 获取OCR信息
                    ocr_info = ocr_scale_info[i] if i < len(ocr_scale_info) else {'scale_length': '', 'unit': ''}
                    
                    results.append({
                        'filename': filenames[i],
                        'true_length_pixels': true_length,
                        'predicted_length_pixels': pred_length,
                        'absolute_error_pixels': absolute_error,
                        'relative_error_percent': relative_error,
                        'ocr_assisted': ocr_assisted[i] if i < len(ocr_assisted) else False,
                        'ocr_scale_length': ocr_info.get('scale_length', ''),
                        'ocr_unit': ocr_info.get('unit', '')
                    })
                    
                    total_error += relative_error
                    num_samples += 1
        
        # 计算总体统计
        mean_error = total_error / num_samples if num_samples > 0 else 0
        
        # 保存结果到CSV
        results_df = pd.DataFrame(results)
        output_file = f'vit_models/{self.model_name}_evaluation_results.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # 打印统计信息
        print(f"📊 评估结果统计:")
        print(f"   总样本数: {num_samples}")
        print(f"   平均相对误差: {mean_error:.2f}%")
        print(f"   最大相对误差: {results_df['relative_error_percent'].max():.2f}%")
        print(f"   最小相对误差: {results_df['relative_error_percent'].min():.2f}%")
        print(f"   标准差: {results_df['relative_error_percent'].std():.2f}%")
        
        # OCR辅助统计
        if use_ocr and OCR_AVAILABLE:
            ocr_assisted_count = results_df['ocr_assisted'].sum()
            ocr_assisted_error = results_df[results_df['ocr_assisted']]['relative_error_percent'].mean()
            non_ocr_error = results_df[~results_df['ocr_assisted']]['relative_error_percent'].mean()
            
            print(f"   OCR辅助样本: {ocr_assisted_count}/{num_samples} ({ocr_assisted_count/num_samples*100:.1f}%)")
            print(f"   OCR辅助平均误差: {ocr_assisted_error:.2f}%")
            print(f"   非OCR辅助平均误差: {non_ocr_error:.2f}%")
        
        print(f"💾 详细结果已保存到: {output_file}")
        
        return results_df, mean_error

def main():
    """主函数"""
    print("🎯 Vision Transformer (ViT) 比例尺检测模型")
    print("="*60)
    print("🔤 集成百度OCR辅助检测")
    print("📏 基于OCR文字区域扩展搜索")
    print("="*60)
    
    # 检查OCR可用性
    if not OCR_AVAILABLE:
        print("⚠️ 百度OCR模块未找到，将使用全图检测模式")
        use_ocr = False
    else:
        print("✅ 百度OCR模块可用，启用OCR辅助检测")
        use_ocr = True
    
    # 创建训练器
    trainer = ViTTrainer(model_name='vit_b_16')
    
    # 创建数据集
    train_dataset, val_dataset = trainer.create_datasets(
        csv_file='scale_bar_labels_fixed.csv',
        img_dir='inputs/lizi/images',
        use_ocr=use_ocr
    )
    
    # 训练模型 - 200轮，不早停
    model = trainer.train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=200,  # 200轮训练
        batch_size=16,
        lr=1e-4
    )
    
    print(f"✅ ViT 模型训练完成!")
    print(f"🎯 最佳长度误差: {trainer.best_length_error:.2f}%")
    if use_ocr:
        print(f"🔤 OCR辅助检测已启用")

if __name__ == '__main__':
    main()
