import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from albumentations.augmentations import transforms

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize

import archs

import losses
from dataset import Dataset, NPZDataset

from metrics import iou_score, indicators, iou_score_s, dice_coef_s, recall_s, precision_s, accuracy_s, jaccard_coef, specificity_s, sensitivity_s, auc_s, f1_score_s, mcc_s

from utils import AverageMeter, str2bool

from tensorboardX import SummaryWriter

import shutil
import os
import subprocess
import os
os.environ['ALBUMENTATIONS_DISABLE_VERSION_CHECK'] = '1'

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--use_Moga', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='busi', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')
    parser.add_argument('--data_format', default='folder', choices=['folder', 'npz'], help='dataset format')
    parser.add_argument('--train_img_npz', type=str, help='path to train image npz file')
    parser.add_argument('--train_mask_npz', type=str, help='path to train mask npz file')
    parser.add_argument('--val_img_npz', type=str, help='path to validation image npz file')
    parser.add_argument('--val_mask_npz', type=str, help='path to validation mask npz file')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

           
            iou = iou_score_s(outputs[-1], target)
            dice = dice_coef_s(outputs[-1], target)
            recall = recall_s(outputs[-1], target)
            precision = precision_s(outputs[-1], target)
            accuracy = accuracy_s(outputs[-1], target)
            jaccard = jaccard_coef(outputs[-1], target)
            specificity = specificity_s(outputs[-1], target)
            sensitivity = sensitivity_s(outputs[-1], target)
            auc = auc_s(outputs[-1], target)
            f1 = f1_score_s(outputs[-1], target)
            mcc = mcc_s(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            
          
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

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', dice),
            ('acc', accuracy),
            ('f1', f1),
            ('auc', auc),
            ('mcc', mcc),
            ('jaccard', jaccard),
            ('sens', sensitivity),
            ('spec', specificity),
            ('prec', precision),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', dice),
                        ('recall', recall),
                        ('precision', precision),
                        ('accuracy', accuracy),
                        ('jaccard', jaccard),
                        ('specificity', specificity),
                        ('sensitivity', sensitivity),
                        ('auc', auc),
                        ('f1', f1),
                        ('mcc', mcc)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter(),
                  'accuracy': AverageMeter(),
                  'auc': AverageMeter(),
                  'f1': AverageMeter(),
                  'mcc': AverageMeter(),
                  'jaccard': AverageMeter(),
                  'sensitivity': AverageMeter(),
                  'specificity': AverageMeter(),
                  'precision': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for i, (input, target, _) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                
                
                iou = iou_score_s(outputs[-1], target)
                dice = dice_coef_s(outputs[-1], target)
                recall = recall_s(outputs[-1], target)
                precision = precision_s(outputs[-1], target)
                accuracy = accuracy_s(outputs[-1], target)
                jaccard = jaccard_coef(outputs[-1], target)
                specificity = specificity_s(outputs[-1], target)
                sensitivity = sensitivity_s(outputs[-1], target)
                auc = auc_s(outputs[-1], target)
                f1 = f1_score_s(outputs[-1], target)
                mcc = mcc_s(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                
                
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

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['accuracy'].update(accuracy, input.size(0))
            avg_meters['auc'].update(auc, input.size(0))
            avg_meters['f1'].update(f1, input.size(0))
            avg_meters['mcc'].update(mcc, input.size(0))
            avg_meters['jaccard'].update(jaccard, input.size(0))
            avg_meters['sensitivity'].update(sensitivity, input.size(0))
            avg_meters['specificity'].update(specificity, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
                ('acc', avg_meters['accuracy'].avg),
                ('auc', avg_meters['auc'].avg),
                ('f1', avg_meters['f1'].avg),
                ('mcc', avg_meters['mcc'].avg),
                ('jaccard', avg_meters['jaccard'].avg),
                ('sens', avg_meters['sensitivity'].avg),
                ('spec', avg_meters['specificity'].avg),
                ('prec', avg_meters['precision'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        ('auc', avg_meters['auc'].avg),
                        ('f1', avg_meters['f1'].avg),
                        ('mcc', avg_meters['mcc'].avg),
                        ('jaccard', avg_meters['jaccard'].avg),
                        ('sensitivity', avg_meters['sensitivity'].avg),
                        ('specificity', avg_meters['specificity'].avg),
                        ('precision', avg_meters['precision'].avg)])

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
    config = vars(parse_args())

    exp_name = config.get('name')
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # create model
    model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], use_Moga=config['use_Moga'])

    model = model.cuda()


    param_groups = []
    for name, param in model.named_parameters():
        if 'kan' in name.lower() and 'fc' in name.lower():
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
        else:
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  

    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
    shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '.png'
    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
 
 
    if config.get('data_format', 'folder') == 'npz':
        train_transform = Compose([
            RandomRotate90(),
            Resize(config['input_h'], config['input_w']),
            # 不加 norm_transform
        ])
        if config['use_Moga']:
            train_transform = Compose([
                RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                Resize(config['input_h'], config['input_w']),
         
            ])
        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
    
        ])
    else:
        RGB_MEAN = (0.485, 0.456, 0.406)
        RGB_STD = (0.229, 0.224, 0.225)
        norm_transform = transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
        train_transform = Compose([
            RandomRotate90(),
            Resize(config['input_h'], config['input_w']),
            norm_transform,
        ])
        if config['use_Moga']:
            train_transform = Compose([
                RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                Resize(config['input_h'], config['input_w']),
                norm_transform,
            ])
        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            norm_transform,
        ])

    # Data loading code
    if config['dataset'] == 'lizi_from_npz':
        RGB_MEAN = (0.485, 0.456, 0.406)
        RGB_STD = (0.229, 0.224, 0.225)
        norm_transform = transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)
        train_transform = Compose([
            RandomRotate90(),
            Resize(config['input_h'], config['input_w']),
            norm_transform,
        ])
        if config['use_Moga']:
            train_transform = Compose([
                RandomRotate90(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                Resize(config['input_h'], config['input_w']),
                norm_transform,
            ])
        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            norm_transform,
        ])
        train_img_dir = os.path.join(config['data_dir'], 'lizi_from_npz/images/train')
        train_mask_dir = os.path.join(config['data_dir'], 'lizi_from_npz/masks/0/train')
        val_img_dir = os.path.join(config['data_dir'], 'lizi_from_npz/images/val')
        val_mask_dir = os.path.join(config['data_dir'], 'lizi_from_npz/masks/0/val')
        train_img_ids = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(train_img_dir)
            if f.endswith('.png') and os.path.splitext(f)[0]
               and os.path.isfile(os.path.join(train_img_dir, f))
               and os.path.isfile(os.path.join(train_mask_dir, os.path.splitext(f)[0] + '_mask.png'))
        ])
        val_img_ids = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(val_img_dir)
            if f.endswith('.png') and os.path.splitext(f)[0]
               and os.path.isfile(os.path.join(val_img_dir, f))
               and os.path.isfile(os.path.join(val_mask_dir, os.path.splitext(f)[0] + '_mask.png'))
        ])
        img_ext = '.png'
        mask_ext = '_mask.png'
    else:
        img_dir = os.path.join(config['data_dir'], config['dataset'], 'images')
        img_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith('.png')])
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])
        img_ext = config.get('img_ext', '.png')
        mask_ext = config.get('mask_ext', '.png')


    if config['dataset'] == 'lizi_from_npz':
        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=train_img_dir,
            mask_dir=train_mask_dir,
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=train_transform
        )
        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=val_img_dir,
            mask_dir=val_mask_dir,
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=val_transform
        )
    else:
        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=img_dir,
            mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=train_transform
        )
        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(config['data_dir'] ,config['dataset'], 'images'),
            mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
            img_ext=img_ext,
            mask_ext=mask_ext,
            num_classes=config['num_classes'],
            transform=val_transform
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('accuracy', []),
        ('auc', []),
        ('f1', []),
        ('mcc', []),
        ('jaccard', []),
        ('sensitivity', []),
        ('specificity', []),
        ('precision', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
        ('val_accuracy', []),
        ('val_auc', []),
        ('val_f1', []),
        ('val_mcc', []),
        ('val_jaccard', []),
        ('val_sensitivity', []),
        ('val_specificity', []),
        ('val_precision', []),
    ])


    best_iou = 0
    best_dice= 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_acc %.4f - val_auc %.4f - val_f1 %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou'], val_log['accuracy'], val_log['auc'], val_log['f1']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['accuracy'].append(train_log['accuracy'])
        log['auc'].append(train_log['auc'])
        log['f1'].append(train_log['f1'])
        log['mcc'].append(train_log['mcc'])
        log['jaccard'].append(train_log['jaccard'])
        log['sensitivity'].append(train_log['sensitivity'])
        log['specificity'].append(train_log['specificity'])
        log['precision'].append(train_log['precision'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        log['val_accuracy'].append(val_log['accuracy'])
        log['val_auc'].append(val_log['auc'])
        log['val_f1'].append(val_log['f1'])
        log['val_mcc'].append(val_log['mcc'])
        log['val_jaccard'].append(val_log['jaccard'])
        log['val_sensitivity'].append(val_log['sensitivity'])
        log['val_specificity'].append(val_log['specificity'])
        log['val_precision'].append(val_log['precision'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('train/dice', train_log['dice'], global_step=epoch)
        my_writer.add_scalar('train/accuracy', train_log['accuracy'], global_step=epoch)
        my_writer.add_scalar('train/auc', train_log['auc'], global_step=epoch)
        my_writer.add_scalar('train/f1', train_log['f1'], global_step=epoch)
        my_writer.add_scalar('train/mcc', train_log['mcc'], global_step=epoch)
        my_writer.add_scalar('train/jaccard', train_log['jaccard'], global_step=epoch)
        my_writer.add_scalar('train/sensitivity', train_log['sensitivity'], global_step=epoch)
        my_writer.add_scalar('train/specificity', train_log['specificity'], global_step=epoch)
        my_writer.add_scalar('train/precision', train_log['precision'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/accuracy', val_log['accuracy'], global_step=epoch)
        my_writer.add_scalar('val/auc', val_log['auc'], global_step=epoch)
        my_writer.add_scalar('val/f1', val_log['f1'], global_step=epoch)
        my_writer.add_scalar('val/mcc', val_log['mcc'], global_step=epoch)
        my_writer.add_scalar('val/jaccard', val_log['jaccard'], global_step=epoch)
        my_writer.add_scalar('val/sensitivity', val_log['sensitivity'], global_step=epoch)
        my_writer.add_scalar('val/specificity', val_log['specificity'], global_step=epoch)
        my_writer.add_scalar('val/precision', val_log['precision'], global_step=epoch)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
    print("=> training finished, best iou: %.4f" % (best_iou))


if __name__ == '__main__':
    main()
