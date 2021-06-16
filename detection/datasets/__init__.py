# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

import os
from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import pathlib
from . import dataset 


def get_dataloader(config):

    trainset = dataset.ImageDataset(
        images_dir=os.path.join('./data', config.project_name, config.train_imgs),
        ann_path=os.path.join('./data', config.project_name, config.train_anns),
        input_size=config.image_size,
        img_channel=3,
        shrink_ratio=0.5,
        train=True,
        transform=transforms.ToTensor()
    )

    valset = dataset.ImageDataset(
        images_dir=os.path.join('./data', config.project_name, config.val_imgs),
        ann_path=os.path.join('./data', config.project_name, config.val_anns),
        input_size=736,
        img_channel=3,
        shrink_ratio=1,
        train=False,
        transform=transforms.ToTensor()
    )

    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True)

    valloader = DataLoader(
        dataset=valset, 
        batch_size=config.batch_size*2, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True)

    trainloader.dataset_len = len(trainset)
    valloader.dataset_len = len(valset)

    return trainloader, valloader
