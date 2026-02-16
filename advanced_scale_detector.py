#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
# import timm  

class AdvancedScaleBarDataset(Dataset):
    
    
    def __init__(self, csv_file, img_dir, transform=None, split='train', model_type='efficientnet'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        self.model_type = model_type
        
        
        self.data['scale_length'] = abs(self.data['x2'] - self.data['x1'])
        
       
        self.data = self.data[self.data['scale_length'] > 0]
        self.data = self.data[self.data['scale_length'] >= 5]
        self.data = self.data[self.data['scale_length'] <= 1000]
        
        print(f" After data filtering: {len(self.data)} samples")
        
        
        self.data = self.data.reset_index(drop=True)
        
        
        self._create_split()
        
        
        self.length_mean = self.data['scale_length'].mean()
        self.length_std = self.data['scale_length'].std()
        print(f"Length statistics: mean={self.length_mean:.2f}, standard deviation={self.length_std:.2f}")
        
    def _create_split(self):
        
        
        short_scales = self.data[self.data['scale_length'] < 100]
        medium_scales = self.data[(self.data['scale_length'] >= 100) & (self.data['scale_length'] < 200)]
        long_scales = self.data[self.data['scale_length'] >= 200]
        
        train_indices = []
        val_indices = []
        
       
        for scale_group in [short_scales, medium_scales, long_scales]:
            if len(scale_group) > 0:
                np.random.seed(42)
                indices = scale_group.index.tolist()
                np.random.shuffle(indices)
                
                train_n = int(len(indices) * 0.8)
                train_indices.extend(indices[:train_n])
                val_indices.extend(indices[train_n:])
        
       
        if self.split == 'train':
            self.indices = train_indices
        else:
            self.indices = val_indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
       
        row_idx = self.indices[idx]
        row = self.data.iloc[row_idx]
        
        
        img_path = os.path.join(self.img_dir, str(row['filename']))
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        if self.transform:
            image = self.transform(image)
        
       
        original_length = abs(row['x2'] - row['x1'])
        
        
        log_length = np.log1p(original_length)
        
        
        normalized_length = (original_length - self.length_mean) / self.length_std
        
        return {
            'image': image,
            'original_length': torch.tensor(original_length, dtype=torch.float32),
            'log_length': torch.tensor(log_length, dtype=torch.float32),
            'normalized_length': torch.tensor(normalized_length, dtype=torch.float32),
            'filename': row['filename']
        }

class EfficientNetScaleDetector(nn.Module):
    
    
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetScaleDetector, self).__init__()
        
       
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1280
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1280
        elif model_name == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1408
        else:
            raise ValueError(f"Unsupported EfficientNet model: {model_name}")
        
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        
        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1)  
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.scale_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        log_length = self.scale_head(features)
        return log_length.squeeze()

class VisionTransformerScaleDetector(nn.Module):
   
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(VisionTransformerScaleDetector, self).__init__()
        
       
        if model_name == 'vit_base_patch16_224':
            self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 768
        elif model_name == 'vit_large_patch16_224':
            self.backbone = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1024
        else:
            raise ValueError(f"Unsupported ViT model: {model_name}")
        
       
        self.backbone.heads = nn.Identity()
        
       
        for param in list(self.backbone.parameters())[:-15]:
            param.requires_grad = False
        
        
        self.scale_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)  
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

class DenseNetScaleDetector(nn.Module):
   
    
    def __init__(self, model_name='densenet121', pretrained=True):
        super(DenseNetScaleDetector, self).__init__()
        
        
        if model_name == 'densenet121':
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1024
        elif model_name == 'densenet169':
            self.backbone = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1664
        elif model_name == 'densenet201':
            self.backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = 1920
        else:
            raise ValueError(f"Unsupported DenseNet model: {model_name}")
        
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        
        for param in list(self.backbone.parameters())[:-25]:
            param.requires_grad = False
        
  
        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 1)  
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.scale_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        log_length = self.scale_head(features)
        return log_length.squeeze()

class AdvancedScaleLengthLoss(nn.Module):
  
    
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super(AdvancedScaleLengthLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta   
        self.gamma = gamma  
        
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred_log_lengths, gt_log_lengths, gt_original_lengths):
       
        log_loss = self.smooth_l1_loss(pred_log_lengths, gt_log_lengths)
        
       
        pred_original_lengths = torch.expm1(pred_log_lengths)
        relative_error = torch.mean(torch.abs(pred_original_lengths - gt_original_lengths) / gt_original_lengths)
        
       
        mse_loss = self.mse_loss(pred_original_lengths, gt_original_lengths)
        
       
        total_loss = self.alpha * log_loss + self.beta * relative_error + self.gamma * mse_loss
        
        return total_loss, log_loss, relative_error

class AdvancedScaleTrainer:
  
    
    def __init__(self, model_type='efficientnet'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.best_length_error = float('inf')
        self.best_model_path = None
        
      
        if model_type == 'efficientnet':
            input_size = 224
            model_name = 'efficientnet_b0'
        elif model_type == 'vit':
            input_size = 224
            model_name = 'vit_base_patch16_224'
        elif model_type == 'densenet':
            input_size = 224
            model_name = 'densenet121'
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.model_name = model_name
        
       
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def create_datasets(self, csv_file, img_dir):
       
        print(f" Creating {self.model_type} dataset...")
        
        train_dataset = AdvancedScaleBarDataset(csv_file, img_dir, self.train_transform, 'train', self.model_type)
        val_dataset = AdvancedScaleBarDataset(csv_file, img_dir, self.val_transform, 'val', self.model_type)
        
        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
        
        return train_dataset, val_dataset
    
    def create_model(self):

        if self.model_type == 'efficientnet':
            model = EfficientNetScaleDetector(model_name=self.model_name, pretrained=True)
        elif self.model_type == 'vit':
            model = VisionTransformerScaleDetector(model_name=self.model_name, pretrained=True)
        elif self.model_type == 'densenet':
            model = DenseNetScaleDetector(model_name=self.model_name, pretrained=True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return model
    
    def train_model(self, train_dataset, val_dataset, epochs=50, batch_size=16, lr=1e-4):
     
        print(f" Start training the {self.model_type} model...")
        print(f"   device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch Size: {batch_size}")
        print(f"   learning_rate: {lr}")
        
       
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        

        model = self.create_model()
        model.to(self.device)
        
   
        criterion = AdvancedScaleLengthLoss(alpha=0.6, beta=0.3, gamma=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
        

        train_losses = []
        val_losses = []
        length_errors = []
        
    
        os.makedirs(f'advanced_{self.model_type}_models', exist_ok=True)
        
        for epoch in range(epochs):
            
            model.train()
            train_loss = 0.0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch in train_pbar:
                images = batch['image'].to(self.device)
                gt_log_lengths = batch['log_length'].to(self.device)
                gt_original_lengths = batch['original_length'].to(self.device)
                
                optimizer.zero_grad()
                
                pred_log_lengths = model(images)
                loss, log_loss, relative_error = criterion(pred_log_lengths, gt_log_lengths, gt_original_lengths)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'RelError': f'{relative_error.item()*100:.2f}%'
                })
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            

            model.eval()
            val_loss = 0.0
            total_length_error = 0.0
            num_samples = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for batch in val_pbar:
                    images = batch['image'].to(self.device)
                    gt_log_lengths = batch['log_length'].to(self.device)
                    gt_original_lengths = batch['original_length'].to(self.device)
                    
                    pred_log_lengths = model(images)
                    loss, log_loss, relative_error = criterion(pred_log_lengths, gt_log_lengths, gt_original_lengths)
                    
                    val_loss += loss.item()
                    total_length_error += relative_error.item() * images.size(0)
                    num_samples += images.size(0)
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'RelError': f'{relative_error.item()*100:.2f}%'
                    })
            
            val_loss /= len(val_loader)
            avg_length_error = total_length_error / num_samples * 100
            val_losses.append(val_loss)
            length_errors.append(avg_length_error)
            

            scheduler.step(val_loss)
            

            if avg_length_error < self.best_length_error:
                self.best_length_error = avg_length_error
                self.best_model_path = f'advanced_{self.model_type}_models/best_{self.model_type}_scale_detector.pth'
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_length_error': self.best_length_error,
                    'model_type': self.model_type
                }, self.best_model_path)
                
                print(f"New best model! Epoch {epoch+1}, length error: {self.best_length_error:.2f}%")
            

            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  training_loss: {train_loss:.4f}")
            print(f"  validation_loss: {val_loss:.4f}")
            print(f"  length_error : {avg_length_error:.2f}%")
            print(f"  best_length_error: {self.best_length_error:.2f}%")
            print("-" * 50)
        

        self._plot_training_curves(train_losses, val_losses, length_errors)
        
        print(f" Training completed! Best model saved to: {self.best_model_path}")
        print(f" best_length_error: {self.best_length_error:.2f}%")
        
        return model
    
    def _plot_training_curves(self, train_losses, val_losses, length_errors):
  
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
    
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Val Loss', color='red')
        ax1.set_title(f'{self.model_type.upper()} Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(length_errors, label='Length Error', color='green')
        ax2.set_title(f'{self.model_type.upper()} Scale Length Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'advanced_{self.model_type}_models/{self.model_type}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
  
    print(" Advanced Scale Ruler Detection Model - Based on Mature Architecture")
    print("="*60)
    
    
    model_types = ['efficientnet', 'vit', 'densenet']
    
    for model_type in model_types:
        print(f"\n Training {model_type.upper()} model...")
        print("="*40)
        
       
        trainer = AdvancedScaleTrainer(model_type=model_type)
        
       
        train_dataset, val_dataset = trainer.create_datasets(
            csv_file='scale_bar_labels_fixed.csv',
            img_dir='inputs/lizi/images'
        )
        

        model = trainer.train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=20, 
            batch_size=16,
            lr=1e-4
        )
        
        print(f"{model_type.upper()} Model training completed!")
        print(f"Best length error: {trainer.best_length_error:.2f}%")
        print("-" * 60)

if __name__ == '__main__':
    main()
