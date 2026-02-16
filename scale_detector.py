#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN Scale Bar Detection Model Implementation
"""
import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Tuple, List, Optional
from kan import KAN, KANLinear

class ScaleDetector(nn.Module):
    """
    Scale Bar Detector using KAN (Kolmogorov-Arnold Networks)
    """
    def __init__(self, 
                 input_size: Tuple[int, int] = (320, 320),
                 hidden_dim: int = 64,
                 grid_size: int = 3,
                 spline_order: int = 2):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        # Feature extraction - increased network depth
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        feature_dim = 128  # Increased feature dimension
        # KAN network for coordinate prediction
        self.kan_coords = KAN(
            layers_hidden=[feature_dim, hidden_dim, 4],
            grid_size=grid_size,
            spline_order=spline_order
        )
        # KANLinear for confidence prediction
        self.kan_conf = nn.Sequential(
            KANLinear(feature_dim, hidden_dim, grid_size=grid_size, spline_order=spline_order),
            nn.ReLU(),
            KANLinear(hidden_dim, 1, grid_size=grid_size, spline_order=spline_order),
            nn.Sigmoid()
        )
        # Fix: Added output activation function to ensure output range
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        feat = self.feature_extractor(x).view(x.size(0), -1)
        coords = self.kan_coords(feat)
        # Fix: Use activation function to ensure output range
        coords = self.output_activation(coords)
        conf = self.kan_conf(feat)
        return coords, conf

class ScaleDetectorWithPreprocessing:
    """
    KAN Scale Bar Detector (with preprocessing)
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
        """
        Preprocess input image for scale bar detection
        
        Args:
            image: Input image (numpy array) in BGR format
            
        Returns:
            Preprocessed tensor ready for model input
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image_resized = cv2.resize(image, self.input_size)
        image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1).unsqueeze(0)
        image_tensor = (image_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
        return image_tensor.to(self.device)
    
    def detect_scale(self, image: np.ndarray, confidence_threshold: float = 0.3):
        """
        Detect scale bar in the input image
        
        Args:
            image: Input image (numpy array) in BGR format
            confidence_threshold: Minimum confidence score for valid detection (default: 0.3)
            
        Returns:
            coords_original: Scale bar coordinates [x1, y1, x2, y2] in original image pixels
            length_original: Scale bar length in original image pixels
            conf: Confidence score of the detection
            Returns None, None, None if detection confidence is below threshold
        """
        with torch.no_grad():
            input_tensor = self.preprocess_image(image)
            scale_coords, confidence = self.model(input_tensor)
            coords = scale_coords[0].cpu().numpy()
            conf = confidence[0].cpu().numpy()[0]
            
            if conf > confidence_threshold:
                h, w = image.shape[:2]
                # Convert normalized coordinates back to pixel coordinates
                x1 = coords[0] * w
                y1 = coords[1] * h
                x2 = coords[2] * w
                y2 = coords[3] * h
                
                # Calculate scale bar length in original image pixels
                length_original = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                coords_original = [int(x1), int(y1), int(x2), int(y2)]
                
                return coords_original, length_original, conf
            else:
                return None, None, None
