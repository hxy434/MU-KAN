import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

class ScaleBarDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        
        # 加载图像
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 创建标签：两个端点的坐标 [x1, y1, x2, y2]
        target = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        
        return image, target, filename

class VGG16UNet(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG16UNet, self).__init__()
        
        # 加载预训练的VGG16
        vgg16 = models.vgg16(pretrained=True)
        self.encoder = vgg16.features
        
        # 编码器特征层
        self.enc1 = self.encoder[:4]    # 64
        self.enc2 = self.encoder[4:9]   # 128
        self.enc3 = self.encoder[9:16]  # 256
        self.enc4 = self.encoder[16:23] # 512
        self.enc5 = self.encoder[23:30] # 512
        
        # 解码器
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(384, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        # 解码器
        dec4 = F.interpolate(enc5, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # 最终输出
        output = self.final(dec1)
        return output

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetUNet, self).__init__()
        
        # 加载预训练的ResNet50
        resnet = models.resnet50(pretrained=True)
        self.encoder = resnet
        
        # 编码器特征层
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 64
        self.enc2 = resnet.layer1  # 256
        self.enc3 = resnet.layer2  # 512
        self.enc4 = resnet.layer3  # 1024
        self.enc5 = resnet.layer4  # 2048
        
        # 解码器 - 修正通道数
        self.dec4 = nn.Sequential(
            nn.Conv2d(2048 + 1024, 512, 3, padding=1),  # 2048 + 1024 = 3072
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1),  # 512 + 512 = 1024
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1),  # 256 + 256 = 512
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),  # 128 + 64 = 192
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        # 解码器
        dec4 = F.interpolate(enc5, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = F.interpolate(dec4, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = F.interpolate(dec3, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = F.interpolate(dec2, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # 最终输出
        output = self.final(dec1)
        return output

class DenseNetUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNetUNet, self).__init__()
        
        # 加载预训练的DenseNet121
        densenet = models.densenet121(pretrained=True)
        
        # 使用完整的DenseNet特征提取器
        self.features = densenet.features
        
        # 解码器 - 使用DenseNet的最终特征
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # 使用完整的DenseNet特征提取
        features = self.features(x)
        
        # 解码器
        decoded = self.decoder(features)
        
        # 最终输出
        output = self.final(decoded)
        return output

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """训练模型"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model.__class__.__name__}_best.pth')
    
    return model

def evaluate_model(model, test_loader, device='cuda'):
    """评估模型并计算相对误差"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, targets, filenames in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            for i, (output, target, filename) in enumerate(zip(outputs, targets, filenames)):
                # 预测的端点坐标
                pred_x1, pred_y1, pred_x2, pred_y2 = output.cpu().numpy()
                true_x1, true_y1, true_x2, true_y2 = target.cpu().numpy()
                
                # 计算像素长度
                pred_length = abs(pred_x2 - pred_x1)
                true_length = abs(true_x2 - true_x1)
                
                # 计算相对误差
                if true_length > 0:
                    relative_error = abs(pred_length - true_length) / true_length * 100
                else:
                    relative_error = 0
                
                results.append({
                    'filename': filename,
                    'pred_x1': pred_x1,
                    'pred_y1': pred_y1,
                    'pred_x2': pred_x2,
                    'pred_y2': pred_y2,
                    'true_x1': true_x1,
                    'true_y1': true_y1,
                    'true_x2': true_x2,
                    'true_y2': true_y2,
                    'predicted_length_pixels': pred_length,
                    'true_length_pixels': true_length,
                    'relative_error_percent': relative_error
                })
    
    return results

# 这个文件只包含模型定义，不包含训练逻辑
# 训练逻辑在 train_scale_bar_models.py 中
