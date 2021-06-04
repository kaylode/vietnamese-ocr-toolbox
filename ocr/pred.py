import os
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import argparse


if __name__ == '__main__':
    config = Cfg.load_config_from_file('/content/repo/final project/src/ocr/train/config.yml')
    config['weights'] = '/content/repo/final project/src/ocr/train/weights/transformerocr.pth'

    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False

    detector = Predictor(config)

    img = '/content/repo/final project/inputs/ocr_data/img/00800.jpg'
    img = Image.open(img)
    plt.imshow(img)
    s = detector.predict(img)
