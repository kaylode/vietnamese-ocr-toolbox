# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import cv2
import torch
import shutil
import numpy as np
import argparse
from tqdm import tqdm
from metrics import _eval
from config import Config

parser = argparse.ArgumentParser('Evaluate PAN')
parser.add_argument('--weight', '-w', type=int, default=10, help='Checkpoint to evaluate')
args = parser.parse_args()


torch.backends.cudnn.benchmark = True


def main(config, args):
    
    pass
    


if __name__ == '__main__':
    config = Config("./config/configs.yaml")
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices

    main(config, args)


    
