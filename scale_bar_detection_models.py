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
        
        # Load image
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create target: coordinates of two endpoints [x1, y1, x2, y2]
        target = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        
        return image, target, filename

class VGG16UNet(nn.Module):
    def __init__(self, num_classes=4):
        super(VGG16UNet, self).__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=True)
        self.encoder = vgg16.features
        
        # Encoder feature layers
        self.enc1 = self.encoder[:4]    # 64 channels
        self.enc2 = self.encoder[4:9]   # 128 channels
        self.enc3 = self.encoder[9:16]  # 256 channels
        self.enc4 = self.encoder[16:23] # 512 channels
        self.enc5 = self.encoder[23:30] # 512 channels
        
        # Decoder layers
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
        
        # Final output layer
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        # Decoder forward pass with skip connections
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
        
        # Final prediction
        output = self.final(dec1)
        return output

class ResNetUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetUNet, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        self.encoder = resnet
        
        # Encoder feature layers
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # 64 channels
        self.enc2 = resnet.layer1  # 256 channels
        self.enc3 = resnet.layer2  # 512 channels
        self.enc4 = resnet.layer3  # 1024 channels
        self.enc5 = resnet.layer4  # 2048 channels
        
        # Decoder layers - corrected channel dimensions
        self.dec4 = nn.Sequential(
            nn.Conv2d(2048 + 1024, 512, 3, padding=1),  # 2048 (enc5) + 1024 (enc4) = 3072 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(512 + 512, 256, 3, padding=1),  # 512 (dec4) + 512 (enc3) = 1024 channels
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, 3, padding=1),  # 256 (dec3) + 256 (enc2) = 512 channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),  # 128 (dec2) + 64 (enc1) = 192 channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final output layer
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Encoder forward pass
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        # Decoder forward pass with skip connections
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
        
        # Final prediction
        output = self.final(dec1)
        return output

class DenseNetUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(DenseNetUNet, self).__init__()
        
        # Load pretrained DenseNet121
        densenet = models.densenet121(pretrained=True)
        
        # Use complete DenseNet feature extractor
        self.features = densenet.features
        
        # Decoder layers - using final DenseNet features
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
        
        # Final output layer
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Complete DenseNet feature extraction
        features = self.features(x)
        
        # Decoder forward pass
        decoded = self.decoder(features)
        
        # Final prediction
        output = self.final(decoded)
        return output

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, device='cuda'):
    """Train the scale bar detection model
    
    Args:
        model: PyTorch model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs (default: 50)
        learning_rate: Initial learning rate (default: 0.001)
        device: Training device ('cuda' or 'cpu', default: 'cuda')
    
    Returns:
        Trained model instance
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
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
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        # Average loss calculation
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{model.__class__.__name__}_best.pth')
    
    return model

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model performance and calculate relative error
    
    Args:
        model: Trained PyTorch model instance
        test_loader: DataLoader for test data
        device: Evaluation device ('cuda' or 'cpu', default: 'cuda')
    
    Returns:
        List of dictionaries containing evaluation results for each sample
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, targets, filenames in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            for i, (output, target, filename) in enumerate(zip(outputs, targets, filenames)):
                # Predicted endpoint coordinates
                pred_x1, pred_y1, pred_x2, pred_y2 = output.cpu().numpy()
                true_x1, true_y1, true_x2, true_y2 = target.cpu().numpy()
                
                # Calculate pixel length of scale bar
                pred_length = abs(pred_x2 - pred_x1)
                true_length = abs(true_x2 - true_x1)
                
                # Calculate relative error percentage
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

# This file only contains model definitions
# Training logic is implemented in train_scale_bar_models.py
