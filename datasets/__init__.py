# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import pathlib
from . import dataset


def get_dataloader(module_name, module_args):

    dataset_args = module_args['dataset']
    dataloader_args = module_args['loader']

    trainset = ImageDataset(
        images_dir=dataset_args['train_imgs'],
        ann_path=dataset_args['train_anns'],
        input_size=dataset_args['input_size'],
        img_channel=dataset_args['img_channel'],
        shrink_ratio=dataset_args['shrink_ratio'],
        transform=transforms.ToTensor()
    )

    trainloader = DataLoader(
        dataset=trainset, 
        batch_size=dataloader_args["batch_size"], 
        shuffle=dataloader_args["shuffle"], 
        num_workers=dataloader_args["num_workers"],
        pin_memory=dataloader_args["pin_memory"])

    trainloader.dataset_len = len(trainset)

    return trainloader
