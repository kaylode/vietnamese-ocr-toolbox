# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 22:00
# @Author  : zhoujun

from __future__ import print_function
import os
from config import Config
from models import get_model, get_loss
from datasets import get_dataloader
from trainer import Trainer
import argparse

parser = argparse.ArgumentParser('Training EfficientDet')
parser.add_argument('config' , type=str, help='project file that contains parameters')
parser.add_argument('--print_per_iter', type=int, default=300, help='Number of iteration to print')
parser.add_argument('--val_interval', type=int, default=2, help='Number of epoches between valing phases')
parser.add_argument('--save_interval', type=int, default=1000, help='Number of steps between saving')
parser.add_argument('--resume', type=str, default=None,
                    help='whether to load weights from a checkpoint, set None to initialize')
parser.add_argument('--saved_path', type=str, default='./weights')
parser.add_argument('--freeze_backbone', action='store_true', help='whether to freeze the backbone')

args = parser.parse_args()
config = Config(os.path.join('configs','config.yaml'))

def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    train_loader = get_dataloader(config)

    criterion = get_loss(config.loss).cuda()

    model = get_model(config.model)

    trainer = Trainer(args=args,
                      config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader)
    trainer.train()


if __name__ == '__main__':
    config = Config("./config/configs.yaml")
    main(config)
