#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：python 
@File ：config_file.py
@Author ：chengbin
@Date ：2022/9/20 12:25 
"""
import argparse
from utils import AverageMeter, str2bool
import archs
import losses

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""
--dataset dataTEM 
--arch UNet
"""

#相当于创建可变变量，为后续训练参数提供选择余地
def parse_args():  # 初始化相关参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')  # 迭代次数epochs
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 16)')  # 批处理参数batch_size

    # model（构建网络模型Unet++）
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')  # 使用的神经网络（该程序使用网络为Unet，当前默认为Unet++）
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=5, type=int,
                        help='input channels')  # 通道数（图像维度，BGR三通道）
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')  # 分类数（该代码只对背景和前景区分，所以为背景+1）
    parser.add_argument('--input_w', default=96, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=96, type=int,
                        help='image height')  # 重置图像尺寸

    # loss（定义损失函数）
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset（定义数据集，包括路径，图像格式）
    parser.add_argument('--dataset', default='dataTEM',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer（定义优化器）
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='optimizer: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')  # 默认使用梯度下降法（SGD）
    #修正了提示命令的笔误
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler（定义学习率，默认即可）
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-4, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--runtime', default='0 min', type=str)
    config = parser.parse_args()
    return config


# config = vars(parse_args())  # 设置默认参数
