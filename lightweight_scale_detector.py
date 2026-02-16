#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级比例尺检测器
适合内存受限环境
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Tuple, List, Optional
from scale_detector import ScaleDetector, ScaleDetectorWithPreprocessing

class LightweightScaleDetector(nn.Module):
    """
    轻量级比例尺检测器
    """
    def __init__(self, 
                 input_size: Tuple[int, int] = (320, 320),
                 hidden_dim: int = 32,
                 grid_size: int = 3,
                 spline_order: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        # 简单CNN特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_coords = nn.Sequential(
            nn.Linear(64, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        self.fc_conf = nn.Sequential(
            nn.Linear(64, hidden_dim//2), nn.ReLU(),
            nn.Linear(hidden_dim//2, 1), nn.Sigmoid()
        )
    def forward(self, x):
        feat = self.feature_extractor(x).view(x.size(0), -1)
        coords = self.fc_coords(feat)
        conf = self.fc_conf(feat)
        return coords, conf

class LightweightScaleDetectorWithPreprocessing:
    """
    轻量级比例尺检测器（带预处理）
    """
    def __init__(self, 
                 model_path: Optional[str] = None,
                 input_size: Tuple[int, int] = (320, 320),
                 hidden_dim: int = 32,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.input_size = input_size
        
        # 使用正确的模型类
        if model_path and 'kan' in model_path:
            # 如果模型文件包含'kan'，使用ScaleDetector
            self.model = ScaleDetector(input_size=input_size, hidden_dim=64)
        else:
            # 否则使用轻量级模型
            self.model = LightweightScaleDetector(input_size=input_size, hidden_dim=hidden_dim)
            
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image_resized = cv2.resize(image, self.input_size)
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)
        # 修复：使用与训练时一致的预处理
        # 训练时：image / 255.0
        # 推理时：也应该使用 image / 255.0
        image_tensor = image_tensor / 255.0
        return image_tensor.to(self.device)
    def detect_scale(self, image: np.ndarray, confidence_threshold: float = 0.3):
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            scale_coords, confidence = self.model(input_tensor)
            coords = scale_coords[0].cpu().numpy()
            conf = confidence[0].cpu().numpy()[0]
            if conf > confidence_threshold:
                h, w = image.shape[:2]
                
                # 修复：使用与训练时一致的输入尺寸归一化
                # 训练时：coords = [x1/640, y1/640, x2/640, y2/640] (输入尺寸归一化)
                # 推理时：先还原到输入尺寸，再映射到原图尺寸
                
                # 1. 模型输出是输入尺寸归一化坐标，还原到输入尺寸
                x1_input = coords[0] * self.input_size[0]
                y1_input = coords[1] * self.input_size[1]
                x2_input = coords[2] * self.input_size[0]
                y2_input = coords[3] * self.input_size[1]
                
                # 2. 映射到原图尺寸
                x1_orig = x1_input * w / self.input_size[0]
                y1_orig = y1_input * h / self.input_size[1]
                x2_orig = x2_input * w / self.input_size[0]
                y2_orig = y2_input * h / self.input_size[1]
                
                coords_original = [int(x1_orig), int(y1_orig), int(x2_orig), int(y2_orig)]
                length_original = np.sqrt((x2_orig - x1_orig) ** 2 + (y2_orig - y1_orig) ** 2)
                
                # 调试输出
                print(f"[修复] 模型输出输入尺寸归一化坐标: {coords}")
                print(f"[修复] 原图尺寸: {w}x{h}")
                print(f"[修复] 输入尺寸坐标: ({x1_input:.1f}, {y1_input:.1f}) -> ({x2_input:.1f}, {y2_input:.1f})")
                print(f"[修复] 原图坐标: {coords_original}")
                print(f"[修复] 像素长度: {length_original:.1f}")
                
                return coords_original, length_original, conf
            else:
                return None, None, None 