import os
import re
from PIL import Image
import numpy as np
from tool.predictor import Predictor
from tool.config import Cfg
import argparse

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def find_rotation_score(img, detector):
    scores = []
    t, score = detector.predict(img, return_prob=True)
    scores.append(score)
    for i in range(3):
        img = img.transpose(Image.ROTATE_90)
        t, score = detector.predict(img, return_prob=True)
        scores.append(score)
    return np.array(scores)

def rotate_img(img, orient):
    for i in range(orient):
        img = img.transpose(Image.ROTATE_90)
    
    return img
    

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
    
