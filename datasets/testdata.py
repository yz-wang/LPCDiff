import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random


class testdata:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=False):
        print("=> evaluating on test set...")

        val_dataset = testDataset(dir=os.path.join(self.config.data.data_dir, 'test_data', 'O_HAZE', 'hazy'),
                                      transforms=self.transforms,
                                      filelist=None,
                                      parse_patches=parse_patches)

        if not parse_patches:
            self.config.sampling.batch_size = 1
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return val_loader


class testDataset(torch.utils.data.Dataset):
    def __init__(self, dir, transforms, filelist=None, parse_patches=True):
        super().__init__()

        if filelist is None:
            test_dir = dir
            input_names = []
            test_inputs = os.path.join(test_dir)
            images = [f for f in listdir(test_inputs) if isfile(os.path.join(test_inputs, f))]

            input_names += [os.path.join(test_inputs, i) for i in images]

            print(len(input_names))
            print(input_names[:5])
        self.dir = dir
        self.input_names = input_names
        self.transforms = transforms
        self.parse_patches = parse_patches

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        img_id = input_name.split('/')[-1].split('.png')[0]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)

        return self.transforms(input_img), img_id


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)