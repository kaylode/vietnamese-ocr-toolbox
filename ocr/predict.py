import os
from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import argparse

parser = argparse.ArgumentParser("Inference PAN")
parser.add_argument('--input', '-i', type=str, help='Path to input image')
parser.add_argument('--output', '-o', type=str, help='Path to folder to save output txt')
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

    if os.path.isfile(args.input):
        img = Image.open(args.input)
        text = detector.predict(img)
        with open(args.output, 'w+') as f:
            f.write(text)

    elif os.path.isdir(args.input):
        img_crop_names = os.listdir(args.input)
        img_crop_names.sort(key=natural_keys)
        crop_texts = []
        for img_crop in img_crop_names:
            img_crop_path = os.path.join(args.input, img_crop)
            img = Image.open(img_crop_path)
            text = detector.predict(img)
            crop_texts.append(text)
        crop_texts = '||'.join(crop_texts)
        
        image_name = os.path.basename(args.input)
        outpath = os.path.join(args.output, image_name, "ocr.txt")
        if not os.path.exists(outpath):
            os.mkdir(outpath)

        with open(outpath, 'w+') as f:
            f.write(crop_texts)

if __name__ == '__main__':
    config = Cfg.load_config_from_file(args.config)
    config['weights'] = args.weight

    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False

    main(config, args)
    
