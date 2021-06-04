import re
import os
from tqdm import tqdm
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import pandas as pd
import argparse

parser = argparse.ArgumentParser("Inference PAN")
parser.add_argument('--input', '-i', type=str, help='Path to input folder containing multicrop images (each in diferrent folder)')
parser.add_argument('--output', '-o', type=str, help='Path to save output csv')
parser.add_argument('--weight', '-w', type=str, help='Path to trained model')
parser.add_argument('--config', '-c', type=str, help='Path to trained model config')
args = parser.parse_args()

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    def atoi(text):
        return int(text) if text.isdigit() else text

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def main(config, args):
    detector = Predictor(config)
    imgnames = os.listdir(args.input)
    extracted_texts = []
    for imgname in tqdm(imgnames):
        img_folder_path = os.path.join(args.input, imgname)
        img_crop_names = os.listdir(img_folder_path)
        img_crop_names.sort(key=natural_keys)
        crop_texts = []
        for img_crop in img_crop_names:
            img_crop_path = os.path.join(img_folder_path, img_crop)
            img = Image.open(img_crop_path)
            text = detector.predict(img)
            crop_texts.append(text)
        crop_texts = '||'.join(crop_texts)
        extracted_texts.append(crop_texts)


    data = {
        'img_name': imgnames,
        'texts': extracted_texts
    }

    df = pd.DataFrame(data, columns= ['img_name', 'texts'])
    df.to_csv(args.output, index=False)

if __name__ == '__main__':
    config = Cfg.load_config_from_file(args.config)
    config['weights'] = args.weight

    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False

    main(config, args)
    
