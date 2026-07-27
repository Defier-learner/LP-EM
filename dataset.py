# -*- coding: utf-8 -*-
"""
@文件名 ：dataset.py
@作者 ：chengbin
@时间 ：2022/5/28
@版本 ：1.0
@测试环境 ：Python3.8(pytorch)
"""
import os
import cv2
import numpy as np
import torch.utils.data
import random
import yaml


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
        # 加载训练使用的默认参数
        with open('config/config.yml', 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        input_channels = config['input_channels']
        img_id = self.img_ids[idx]
        img_number = int(img_id[5:])
        img_all = []
        mask_all = []
        for input_channel in range(0, input_channels):
            img_number_new = img_number - int(input_channels/2) + input_channel
            img_path = os.path.join(self.img_dir, img_id[0:5] + str(img_number_new) + self.img_ext)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
            else:
                img_path = os.path.join(self.img_dir, img_id + self.img_ext)
                img = cv2.imread(img_path)
        #取邻域图，尽管是按通道数取出的领域图，但取图时是取得完整通道

            mask = []
            mask_path = os.path.join(self.mask_dir, '0', img_id[0:5] + str(img_number_new) + self.mask_ext)#修改了原扩展名格式笔误，尽管不影响程序运行
            #‘0’是因为二分类问题直接写死，实际应根据目录分类数进行确定
            if os.path.exists(mask_path):
                mask.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None])
            else:
                mask_path = os.path.join(self.mask_dir, '0', img_id + self.mask_ext)
                mask.append(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None])
            mask = mask[0]#代码冗余，不用先放进去再取出来

            if self.transform is not None:
                augmented = self.transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            a = random.randint(0, 2)
            img = img[:, :, a]
            img_all.append(img)
            mask_all.append(mask)
        img = np.dstack(img_all)
        mask = mask_all[int(input_channel/2)]
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
