# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import pathlib
from . import dataset 


def get_dataloader(config):

    trainset = dataset.ImageDataset(
        images_dir=config.train_imgs,
        ann_path=config.train_anns,
        input_size=config.image_size,
        img_channel=3,
        shrink_ratio=0.5,
        transform=transforms.ToTensor()
    )

    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True)

    trainloader.dataset_len = len(trainset)

    return trainloader
