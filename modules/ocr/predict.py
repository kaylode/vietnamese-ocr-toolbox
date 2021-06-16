import os
import re
from PIL import Image
import cv2
import numpy as np
from ocr.tool.predictor import Predictor
from ocr.tool.config import Cfg
from tool.utils import natural_keys
import argparse



def find_rotation_score(img, detector):
    scores = []
    t, score = detector(img, return_prob=True)
    scores.append(score)
    new_img = img.copy()
    for i in range(3):
        new_img = cv2.rotate(new_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        t, score = detector(new_img, return_prob=True)
        scores.append(score)
    return np.array(scores)

def rotate_img(img, orient):
    new_img = img.copy()
    for i in range(orient):
        new_img = cv2.rotate(new_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_img
    

def main(config, args):
    detector = Predictor(config)

    if os.path.isfile(args.input):
        img = Image.open(args.input)
        best_orient = find_best_rotation(img, detector)
        img = rotate_img(img, best_orient)
        text = detector.predict(img)
        with open(args.output, 'w+') as f:
            f.write(text)

    elif os.path.isdir(args.input):
        img_crop_names = os.listdir(args.input)
        img_crop_names.sort(key=natural_keys)
        crop_texts = []
        for i, img_crop in enumerate(img_crop_names):
            img_crop_path = os.path.join(args.input, img_crop)
            img = Image.open(img_crop_path)
            if i == 0:
                best_orient = find_best_rotation(img, detector)
                print(f"Rotate image by {best_orient*90} degrees")
            img = rotate_img(img, best_orient)
            text = detector.predict(img)
            crop_texts.append(text)
        crop_texts = '||'.join(crop_texts)
        
        with open(args.output, 'w+') as f:
            f.write(crop_texts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Inference PAN")
    parser.add_argument('--input', '-i', type=str, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, help='Path to save output txt')
    parser.add_argument('--weight', '-w', type=str, help='Path to trained model')
    parser.add_argument('--config', '-c', type=str, help='Path to trained model config')
    args = parser.parse_args()

    config = Cfg.load_config_from_file(args.config)
    config['weights'] = args.weight

    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False

    main(config, args)
    
