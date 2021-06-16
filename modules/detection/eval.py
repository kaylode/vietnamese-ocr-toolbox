# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import cv2
import torch
import shutil
import numpy as np
import argparse
from config import Config
from predict import PAN
from metrics import mAPScores

parser = argparse.ArgumentParser('Evaluate PAN')
parser.add_argument('--weight', '-w', type=int, default=10, help='Checkpoint to evaluate')
args = parser.parse_args()


torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    config = Config("./config/configs.yaml")
    metric = mAPScores(
        ann_file= os.path.join('../data', config.project_name, config.val_anns),
        img_dir=os.path.join('../data', config.project_name, config.val_imgs)
    )

    model = PAN(config, model_path=args.weight)

    metric.update(model)    
    print(metric.value())
    
