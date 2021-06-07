import os
import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from preprocess import DocScanner
import detection
import ocr

parser = argparse.ArgumentParser("Document Extraction")
parser.add_argument("--input", help="Path to single image to be scanned")
parser.add_argument("--output", default="./results", help="Path to output folder")
args = parser.parse_args()


PREPROCESS_RES=f"{args.output}/preprocessed.jpg"
DETECTION_RES=f"{args.output}/detected.jpg"
DETECTION_FOLDER_RES=f"{args.output}/crops"
OCR_RES=f"{args.output}/ocr.txt"

PAN_WEIGHT="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/detection-checkpoints/PANNet_best.pth"
OCR_WEIGHT="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/ocr-checkpoints/transformerocr.pth"
OCR_CONFIG="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/ocr-checkpoints/config.yml"


if __name__ == "__main__":

    # Document extraction
    scanner = DocScanner()
    os.makedirs(args.output,exist_ok=True)
    scanner.scan(args.input, PREPROCESS_RES)

    # Text detection model + OCR model config
    det_config = detection.Config(os.path.join('detection', 'config','configs.yaml'))
    os.environ['CUDA_VISIBLE_DEVICES'] = det_config.gpu_devices
    ocr_config = ocr.Cfg.load_config_from_file(OCR_CONFIG)
    ocr_config['weights'] = OCR_WEIGHT
    ocr_config['cnn']['pretrained']=False
    ocr_config['device'] = 'cuda:0'
    ocr_config['predictor']['beamsearch']=False

    det_model = detection.PAN(det_config, model_path=PAN_WEIGHT)
    ocr_model = ocr.Predictor(ocr_config)

    # Find best rotation by forwarding one pioneering image and calculate the score for each orientation
    TOP_K = 5

    preds, boxes_list, t = det_model.predict(
        PREPROCESS_RES, 
        DETECTION_FOLDER_RES, 
        crop_region=True,
        num_boxes=TOP_K)

    orientation_scores = np.array([0.,0.,0.,0.])
    for i in range(TOP_K):
        img = Image.open(os.path.join(DETECTION_FOLDER_RES, f'{i}.jpg'))
        orientation_scores += ocr.find_rotation_score(img, ocr_model)
    best_orient = np.argmax(orientation_scores)
    print(f"Rotate image by {best_orient*90} degrees")

    # Rotate the original image
    rotated_img = ocr.rotate_img(Image.open(PREPROCESS_RES), best_orient)
    rotated_img.save(PREPROCESS_RES)

    # Detect and OCR for final result
    preds, boxes_list, t = det_model.predict(
        PREPROCESS_RES, 
        DETECTION_FOLDER_RES, 
        crop_region=True)
    
    detection.show_img(preds)
    img = detection.draw_bbox(cv2.imread(PREPROCESS_RES)[:, :, ::-1], boxes_list)
    detection.show_img(img, color=True)
    plt.axis('off')
    plt.savefig(DETECTION_RES,bbox_inches='tight')

    # OCR
    img_crop_names = os.listdir(DETECTION_FOLDER_RES)
    img_crop_names.sort(key=ocr.natural_keys)
    crop_texts = []
    for i, img_crop in enumerate(img_crop_names):
        img_crop_path = os.path.join(DETECTION_FOLDER_RES, img_crop)
        img = Image.open(img_crop_path)
        text = ocr_model.predict(img)
        crop_texts.append(text)
    crop_texts = '||'.join(crop_texts)
    
    with open(OCR_RES, 'w+') as f:
        f.write(crop_texts)