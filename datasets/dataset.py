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

        train_dataset = AllWeatherDataset(dir=os.path.join(self.config.data.data_dir, 'train_data', 'NH_HAZE'),
                                          patch_size=self.config.data.patch_size)
                                          # filelist='{}_train.txt'.format(self.config.data.train_dataset))
        # val_dataset = AllWeatherDataset(os.path.join(self.config.data.data_dir, self.config.data.val_dataset, 'val'),
        #                                 patch_size=self.config.data.patch_size,
        #                                 filelist='{}_val.txt'.format(self.config.data.val_dataset), train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
        #                                          num_workers=self.config.data.num_workers,
        #                                          pin_memory=True)

        return train_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, train=True):
        super().__init__()
        self.train = train

        ITS_dir = dir
        input_names, gt_names = [], []
        ITS_inputs = os.path.join(ITS_dir, 'hazy')
        images = [f for f in listdir(ITS_inputs) if isfile(os.path.join(ITS_inputs, f))]  # and f[:2] not in [str(i) for i in range(51, 56)] and "GT" not in f]

        input_names += [os.path.join(ITS_inputs, i) for i in images]
        # gt_names += [os.path.join(os.path.join(ITS_dir, 'clean'), i.split('/')[-1].split('_')[0] + '.png') for i in
        #              input_names]
        gt_names += [os.path.join(os.path.join(ITS_dir, 'clean'), i.split('/')[-1].replace('hazy', 'GT')) for i in
                     input_names]  # i.replace('hazy', 'GT')) for i in images]
        # gt_names += [os.path.join(os.path.join(ITS_dir, 'clean'), i.split('/')[-1]) for i in
        #              input_names]
        print(input_names[0])
        print(gt_names[0])
        print(len(input_names))
        self.dir = None
        # x = list(enumerate(input_names))  # enumerate函数为input_names列表中的每个文件名添加索引，并将列表中的元素顺序打乱。
        # random.shuffle(x)
        # indices, input_names = zip(*x)
        # gt_names = [gt_names[idx] for idx in indices]
        # self.file_list = filelist
        # self.train_list = os.path.join(dir, self.file_list)
        # with open(self.train_list) as f:
        #     contents = f.readlines()
        #     input_names = [i.strip() for i in contents]
        #     gt_names = [i.strip().replace('low', 'high') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        # self.transforms = PairCompose([
        #         PairToTensor()
        #     ])
        if self.train:
            self.transforms = PairCompose([
                PairRandomCrop(self.patch_size),
                PairToTensor()
            ])
        else:
            self.transforms = PairCompose([
                PairToTensor()
            ])

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        # img_id = re.split('/', input_name)[-1].split('_')[0]
        img_id = re.split('/', input_name)[-1].split('_')[0]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        # gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        input_img, gt_img = self.transforms(input_img, gt_img)

        return torch.cat([input_img, gt_img], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
