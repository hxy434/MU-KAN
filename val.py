#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import OrderedDict

import archs

from dataset import Dataset, NPZDataset
from metrics import iou_score, iou_score_s, dice_coef_s, recall_s, precision_s, accuracy_s, jaccard_coef, specificity_s, sensitivity_s, auc_s, f1_score_s, mcc_s
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    parser.add_argument('--test', action='store_true', help='run on test set if available (for npz or folder dataset)')
    parser.add_argument('--test_img_dir', type=str, default=None, help='test images dir (for folder format)')
    parser.add_argument('--test_mask_dir', type=str, default=None, help='test masks dir (for folder format)')

    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']] (
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        use_Moga=config['use_Moga']
    )
    model = model.cuda()

    # 自动切换数据集类型
    if config.get('data_format', 'folder') == 'npz':
        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            # 不加 transforms.Normalize()
        ])
        # 判断是否需要测试集
        if args.test and os.path.exists(config['val_img_npz'].replace('val', 'test')) and os.path.exists(config['val_mask_npz'].replace('val', 'test')):
            print('==> Running on TEST set')
            test_img_npz = config['val_img_npz'].replace('val', 'test')
            test_mask_npz = config['val_mask_npz'].replace('val', 'test')
            test_dataset = NPZDataset(
                img_npz=test_img_npz,
                mask_npz=test_mask_npz,
                transform=val_transform
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                drop_last=False)
            def safe_get_img_id(meta):
                if isinstance(meta, dict):
                    return meta.get('idx', meta.get('img_id', str(meta)))
                return str(meta)
            get_img_id = safe_get_img_id
            loader = test_loader
            out_dir = os.path.join(args.output_dir, config['name'], 'out_test')
        else:
            val_dataset = NPZDataset(
                img_npz=config['val_img_npz'],
                mask_npz=config['val_mask_npz'],
                transform=val_transform
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                drop_last=False)
            def safe_get_img_id(meta):
                if isinstance(meta, dict):
                    return meta.get('idx', meta.get('img_id', str(meta)))
                return str(meta)
            get_img_id = safe_get_img_id
            loader = val_loader
            out_dir = os.path.join(args.output_dir, config['name'], 'out_val')
    else:
        dataset_name = config['dataset']
        img_ext = '.png'
        mask_ext = '.png'
        if dataset_name == 'busi' or dataset_name == 'lizi_from_npz':
            mask_ext = '_mask.png'
        elif dataset_name == 'glas':
            mask_ext = '.png'
        if args.test and args.test_img_dir and args.test_mask_dir:
            print('==> Running on TEST set (folder format)')
            test_img_dir = args.test_img_dir
            test_mask_dir = args.test_mask_dir
            test_img_ids = []
            for f in os.listdir(test_img_dir):
                if f.endswith(img_ext) and os.path.splitext(f)[0]:
                    img_id = os.path.splitext(f)[0]
                    img_path = os.path.join(test_img_dir, f)
                    mask_path = os.path.join(test_mask_dir, img_id + mask_ext)
                    if os.path.isfile(img_path) and os.path.isfile(mask_path):
                        test_img_ids.append(img_id)
                    else:
                        if not os.path.isfile(mask_path):
                            print(f'[WARN] mask not found for: {img_id}, expected: {mask_path}')
            print(f'[INFO] test_img_ids loaded: {test_img_ids}')
            if not test_img_ids:
                print('[ERROR] No valid test images found! Please check your test_img_dir and test_mask_dir.')
            val_transform = Compose([
                Resize(config['input_h'], config['input_w']),
                transforms.Normalize(),
            ])
            test_dataset = Dataset(
                img_ids=test_img_ids,
                img_dir=test_img_dir,
                mask_dir=test_mask_dir,
                img_ext=img_ext,
                mask_ext=mask_ext,
                num_classes=config['num_classes'],
                transform=val_transform)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                drop_last=False)
            def safe_get_img_id(meta):
                if isinstance(meta, dict):
                    return meta.get('img_id', meta.get('idx', str(meta)))
                return str(meta)
            get_img_id = safe_get_img_id
            loader = test_loader
            out_dir = os.path.join(args.output_dir, config['name'], 'out_test')
        else:
            img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
            img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
            _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])
            val_transform = Compose([
                Resize(config['input_h'], config['input_w']),
                transforms.Normalize(),
            ])
            val_dataset = Dataset(
                img_ids=val_img_ids,
                img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
                mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
                img_ext=img_ext,
                mask_ext=mask_ext,
                num_classes=config['num_classes'],
                transform=val_transform)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                drop_last=False)
            def safe_get_img_id(meta):
                if isinstance(meta, dict):
                    return meta.get('img_id', meta.get('idx', str(meta)))
                return str(meta)
            get_img_id = safe_get_img_id
            loader = val_loader
            out_dir = os.path.join(args.output_dir, config['name'], 'out_val')

    # ====== 验证数据调试代码 ======

    ckpt = torch.load(f'{args.output_dir}/{args.name}/model.pth')

    try:
        model.load_state_dict(ckpt)
    except Exception as e:
        print("Pretrained model keys:", ckpt.keys())
        print("Current model keys:", model.state_dict().keys())
        print("Difference in model keys:")
        print("模型有但权重没有:", set(model.state_dict().keys()) - set(ckpt.keys()))
        print("权重有但模型没有:", set(ckpt.keys()) - set(model.state_dict().keys()))
        print("Exception:", e)
        model.load_state_dict(ckpt, strict=False)

    model.eval()

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    hd95_avg_meter = AverageMeter()
    accuracy_avg_meter = AverageMeter()
    auc_avg_meter = AverageMeter()
    f1_avg_meter = AverageMeter()
    mcc_avg_meter = AverageMeter()
    jaccard_avg_meter = AverageMeter()
    sensitivity_avg_meter = AverageMeter()
    specificity_avg_meter = AverageMeter()
    precision_avg_meter = AverageMeter()

    print(f'验证集总batch数: {len(loader)}')
    with torch.no_grad():
        for batch_idx, (input, target, meta) in enumerate(tqdm(loader, total=len(loader), desc="Validating")):
            if batch_idx == 0:  # 只在第一个batch打印meta信息
                print(f'当前batch {batch_idx} meta: {meta}')
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            # 使用独立的指标计算函数
            iou = iou_score_s(output, target)
            dice = dice_coef_s(output, target)
            recall = recall_s(output, target)
            precision = precision_s(output, target)
            accuracy = accuracy_s(output, target)
            jaccard = jaccard_coef(output, target)
            specificity = specificity_s(output, target)
            sensitivity = sensitivity_s(output, target)
            auc = auc_s(output, target)
            f1 = f1_score_s(output, target)
            mcc = mcc_s(output, target)
            
            # 更新所有指标的平均值
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            accuracy_avg_meter.update(accuracy, input.size(0))
            auc_avg_meter.update(auc, input.size(0))
            f1_avg_meter.update(f1, input.size(0))
            mcc_avg_meter.update(mcc, input.size(0))
            jaccard_avg_meter.update(jaccard, input.size(0))
            sensitivity_avg_meter.update(sensitivity, input.size(0))
            specificity_avg_meter.update(specificity, input.size(0))
            precision_avg_meter.update(precision, input.size(0))

            # 更新进度条显示
            if batch_idx % 10 == 0:  # 每10个batch更新一次显示
                tqdm.write(f'Batch {batch_idx}: IoU={iou:.4f}, Dice={dice:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}')

            output = torch.sigmoid(output).cpu().numpy()
            output[output>=0.5]=1
            output[output<0.5]=0

            os.makedirs(out_dir, exist_ok=True)
            # 修复meta为dict且img_id为list时的保存逻辑
            if isinstance(meta, dict) and 'img_id' in meta:
                img_ids = meta['img_id']
            else:
                img_ids = meta
            for pred, img_id in zip(output, img_ids):
                if isinstance(img_id, (list, tuple)):
                    img_id = str(img_id[0])
                pred_np = pred[0].astype(np.uint8)
                pred_np = pred_np * 255
                img = Image.fromarray(pred_np, 'L')
                img.save(os.path.join(out_dir, f'{img_id}.jpg'))

    print(config['name'])
    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)
    print('Accuracy: %.4f' % accuracy_avg_meter.avg)
    print('AUC: %.4f' % auc_avg_meter.avg)
    print('F1 Score: %.4f' % f1_avg_meter.avg)
    print('MCC: %.4f' % mcc_avg_meter.avg)
    print('Jaccard: %.4f' % jaccard_avg_meter.avg)
    print('Sensitivity: %.4f' % sensitivity_avg_meter.avg)
    print('Specificity: %.4f' % specificity_avg_meter.avg)
    print('Precision: %.4f' % precision_avg_meter.avg)

if __name__ == '__main__':
    main()