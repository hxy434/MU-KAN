#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN比例尺检测模型实现
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Tuple, List, Optional
from kan import KAN, KANLinear

class ScaleDetector(nn.Module):
    """
    使用KAN网络的比例尺检测器
    """
    def __init__(self, 
                 input_size: Tuple[int, int] = (320, 320),
                 hidden_dim: int = 64,
                 grid_size: int = 3,
                 spline_order: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        # 特征提取 - 增加网络深度
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        feature_dim = 128  # 增加特征维度
        # KAN网络用于坐标预测
        self.kan_coords = KAN(
            layers_hidden=[feature_dim, hidden_dim, 4],
            grid_size=grid_size,
            spline_order=spline_order
        )
        # KANLinear用于置信度预测
        self.kan_conf = nn.Sequential(
            KANLinear(feature_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order),
            nn.ReLU(),
            KANLinear(hidden_dim, 1, grid_size=grid_size, spline_order=spline_order),
            nn.Sigmoid()
        )
        # 修复：添加输出激活函数，确保输出范围
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        feat = self.feature_extractor(x).view(x.size(0), -1)
        coords = self.kan_coords(feat)
        # 修复：使用激活函数确保输出范围
        coords = self.output_activation(coords)
        conf = self.kan_conf(feat)
        return coords, conf

class ScaleDetectorWithPreprocessing:
    """
    KAN比例尺检测器（带预处理）
    """
    def __init__(self, 
                 model_path: Optional[str] = None,
                 input_size: Tuple[int, int] = (320, 320),
                 hidden_dim: int = 64,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.input_size = input_size
        self.model = ScaleDetector(input_size=input_size, hidden_dim=hidden_dim)
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
        image_tensor = (image_tensor / 255.0 - 0.5) * 2
        return image_tensor.to(self.device)
    def detect_scale(self, image: np.ndarray, confidence_threshold: float = 0.3):
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            scale_coords, confidence = self.model(input_tensor)
            coords = scale_coords[0].cpu().numpy()
            conf = confidence[0].cpu().numpy()[0]
            if conf > confidence_threshold:
                h, w = image.shape[:2]
                # 归一化还原为像素坐标
                x1 = coords[0] * w
                y1 = coords[1] * h
                x2 = coords[2] * w
                y2 = coords[3] * h
                length_original = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                coords_original = [int(x1), int(y1), int(x2), int(y2)]
                return coords_original, length_original, conf
            else:
                return None, None, None 