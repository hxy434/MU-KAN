import os

import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            # 只针对 lizi_from_npz，不加类别子目录
            if self.mask_dir and 'lizi_from_npz' in self.mask_dir:
                mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
            else:
                mask_path = os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)
            binary = np.array(Image.open(mask_path).convert('L'))
            mask.append(binary)
        mask = np.dstack(mask)
        # print(f'first: {binary.max()}')

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        # print(f'before: {mask.max()}')
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        # print(f'after: {mask.max()}')
        if mask.max()<1:
            mask[mask>0] = 1.0

        return img, mask, {'img_id': img_id}

class NPZDataset(torch.utils.data.Dataset):
    """
    用于加载 npz 或 npy 格式的医学分割数据集。
    """
    def __init__(self, img_npz, mask_npz, transform=None, img_key=None, mask_key=None):
        # 自动兼容 npy 和 npz
        img_npz_obj = np.load(img_npz)
        mask_npz_obj = np.load(mask_npz)
        if isinstance(img_npz_obj, np.ndarray):
            self.images = img_npz_obj
        else:
            if img_key is None:
                img_key = list(img_npz_obj.keys())[0]
            self.images = img_npz_obj[img_key]
        if isinstance(mask_npz_obj, np.ndarray):
            self.masks = mask_npz_obj
        else:
            if mask_key is None:
                mask_key = list(mask_npz_obj.keys())[0]
            self.masks = mask_npz_obj[mask_key]
        self.transform = transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = self.images[idx]  # shape: H, W 或 H, W, 1
        mask = self.masks[idx]  # shape: H, W 或 H, W, 1
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        img = img.astype('float32') / 255 if img.max() > 1 else img.astype('float32')
        if img.ndim == 2:
            img = img[None, ...]  # (1, H, W)
        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)  # (3, H, W)
        mask = mask.astype('float32')
        if mask.ndim == 2:
            mask = mask[None, ...]  # (1, H, W)
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = np.transpose(mask, (2, 0, 1))  # (1, H, W)
        print(f'NPZDataset 返回 meta: {idx}')
        return img, mask, str(idx)
