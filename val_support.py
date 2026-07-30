#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：python 
@File ：val_support.py
@Author ：chengbin
@Date ：2022/9/20 15:46 
"""
import argparse
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
# pip install PyYaml
import archs
import albumentations as A
# https://github.com/albumentations-team/albumentations
# pip install -U albumentations
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm
import shutil
from glob import glob


# 初始化相关参数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="dataTEM_NestedUNet_woDS",
                        help='model name')
    args = parser.parse_args()
    return args


def support(image_path):
    if os.path.exists('config'):
        pass
    else:
        os.mkdir('config')
    if os.path.isfile('config/config.yml'):
        os.remove('config/config.yml')
        shutil.copy('models/dataTEM_NestedUNet_woDS/config.yml', 'config')
    else:
        shutil.copy('models/dataTEM_NestedUNet_woDS/config.yml', 'config')

    if os.path.exists('outputs'):
        shutil.rmtree('outputs')
    else:
        pass
    if os.path.exists('predict'):
        shutil.rmtree('predict')
        os.mkdir('predict')
    else:
        os.mkdir('predict')
    os.mkdir('predict/dataTEM')
    os.mkdir('predict/dataTEM/images')
    os.mkdir('predict/dataTEM/masks')
    os.mkdir('predict/dataTEM/masks/0')
    names = os.listdir(image_path)
    for name in names:
        file_path = os.path.join(image_path, name)
        if os.path.isfile(file_path):  # 用于判断某一对象(需提供绝对路径)是否为文件
            shutil.copy(file_path, 'predict/dataTEM/images')
            shutil.copy(file_path, 'predict/dataTEM/masks/0')
            #把原始图片放到mask里，是为了允许计算dice_loss，尽管该loss没有意义，但目的是得到预测图

    args = parse_args()  # 设置默认参数

    # 加载训练使用的默认参数
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-' * 20)

    # 构建模型
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    model = model.cuda()

    # 加载数据
    img_ids = glob(os.path.join('predict', config['dataset'], 'images', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    # 验证集数据
    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('predict', config['dataset'], 'images'),
        mask_dir=os.path.join('predict', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # 计算输出
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)
            iou = iou_score(output, target)
            #用来生成可视化报告？？？？？？
            avg_meter.update(iou, input.size(0))
            output = torch.sigmoid(output).cpu().numpy()
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))
    # print('IoU: %.4f' % avg_meter.avg)
    # plot_examples(input, target, model, num_examples=3)
    torch.cuda.empty_cache()
    shutil.rmtree('predict')
    return config
