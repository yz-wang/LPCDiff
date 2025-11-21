import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torch.utils.data
import PIL
from PIL import Image
import re
import random

from datasets.data_augment import PairCompose, PairRandomCrop, PairToTensor


class LLdataset:
    def __init__(self, config):
        self.config = config

    def get_loaders(self):

        train_dataset = AllWeatherDataset(dir=os.path.join(self.config.data.data_dir, 'train_data', 'O_HAZE'),
                                          patch_size=self.config.data.patch_size)

        val_dataset = AllWeatherDataset(dir=os.path.join(self.config.data.data_dir, 'test_data', 'O_HAZE'),
                                          patch_size=None, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                                 num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, train=True):
        super().__init__()
        self.train = train

        ITS_dir = dir
        input_names, gt_names = [], []
        ITS_inputs = os.path.join(ITS_dir, 'hazy')
        images = [f for f in listdir(ITS_inputs) if isfile(os.path.join(ITS_inputs, f))]

        input_names += [os.path.join(ITS_inputs, i) for i in images]


        gt_names += [os.path.join(os.path.join(ITS_dir, 'gt'), i.split('/')[-1].replace('hazy', 'GT')) for i in
                     input_names]
        print((input_names[:5]))
        print(gt_names[:5])
        print(len(input_names))
        self.dir = None

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size

        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
        ])
        else:  # 验证时使用整图
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1].split('_')[0]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id
    # def get_images(self, index):
    #     input_name = self.input_names[index]
    #     gt_name = self.gt_names[index]
    #     img_id = re.split('/', input_name)[-1].split('_')[0]
    #
    #     # 强制都转成RGB，确保3通道一致
    #     input_img = PIL.Image.open(input_name).convert('RGB')
    #     gt_img = PIL.Image.open(gt_name).convert('RGB')
    #
    #     # 应用双图像变换
    #     input_img, gt_img = self.transforms(input_img, gt_img)
    #
    #     # 拼接成 [6, H, W]
    #     x = torch.cat([input_img, gt_img], dim=0)  # [3+3, H, W]
    #     return x, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
