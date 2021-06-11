import os
import cv2
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from preprocess import DocScanner
import detection
import ocr
import retrieval

parser = argparse.ArgumentParser("Document Extraction")
parser.add_argument("--input", help="Path to single image to be scanned")
parser.add_argument("--output", default="./results", help="Path to output folder")
args = parser.parse_args()


PREPROCESS_RES=f"{args.output}/preprocessed.jpg"
DETECTION_RES=f"{args.output}/detected.jpg"
DETECTION_CSV_RES=f"{args.output}/box_info.csv"
DETECTION_FOLDER_RES=f"{args.output}/crops"
OCR_RES=f"{args.output}/ocr.txt"
FINAL_RES=f"{args.output}/final.jpg"

PAN_WEIGHT="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/detection-checkpoints/PANNet_best.pth"
OCR_WEIGHT="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/ocr-checkpoints/transformerocr.pth"
OCR_CONFIG="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/ocr-checkpoints/config.yml"
BERT_WEIGHT="/content/drive/MyDrive/AI Competitions/MC-OCR/checkpoints/retrieval-checkpoints/phobert_report.pth"

LABEL_TO_IDX = {"SELLER":0, "ADDRESS":1, "TIMESTAMP":2, "TOTAL_COST":3, "NONE":4}
IDX_TO_LABEL = {0: "SELLER", 1: "ADDRESS", 2: "TIMESTAMP", 3: "TOTAL_COST", 4:"NONE"}

def visualize(img, boxes, texts, labels, probs, img_name):
    """
    Visualize an image with its bouding boxes
    """
    STANDARD_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,0,0)]

    dpi = matplotlib.rcParams['figure.dpi']
    # Determine the figures size in inches to fit your image
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)

    def find_highest_score_each_class(labels, probs):
        best_score = [0,0,0,0]
        best_idx = [-1,-1,-1,-1]
        for i, (label, prob) in enumerate(zip(labels, probs)):
            label_idx = LABEL_TO_IDX[label]
            if label_idx != 4:
                if prob > best_score[label_idx]:
                    best_score[label_idx] = prob
                    best_idx[label_idx] = i
        return best_idx
    
    best_score_idx = find_highest_score_each_class(labels, probs)
    fig,ax = plt.subplots(figsize=figsize)
    
    
    # Create a Rectangle patch
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i, (box,text,label,prob) in enumerate(zip(boxes,texts,labels,probs)):
        label_idx = LABEL_TO_IDX[label]
        color = STANDARD_COLORS[label_idx]
        box = eval(box)
        x1,y1,x2,y2,x3,y3,x4,y4 = box
        box = np.array([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
        img = detection.draw_bbox(img, [box], color=color)
        score = np.round(float(prob), 3)

        if i in best_score_idx:
            plt_text = f'{text}: {label} | {score}'
            plt.text(x1, y1-3, plt_text, color = [i/255 for i in color], fontsize=10, weight="bold")

    # Display the image
    
    ax.imshow(img)

    plt.axis('off')
    plt.savefig(img_name,bbox_inches='tight')
    plt.close()

def merge_result(df):
    preds = []
    probs = []

    for id, row in df.iterrows():
        if row["timestamp"] == 1:
            preds.append("TIMESTAMP")
            probs.append(5.0)
        elif row["bert_labels"] == row["diff_labels"]:
            preds.append(row["bert_labels"])
            probs.append(row["bert_probs"] + row["diff_probs"])
        elif row["bert_labels"] == row["trie_labels"]:
            preds.append(row["bert_labels"])
            probs.append(row["bert_probs"] + row["trie_probs"])
        elif row["trie_labels"] == row["diff_labels"]:
            preds.append(row["trie_labels"])
            probs.append(row["trie_probs"] + row["diff_probs"])
        else:
            if row["diff_probs"] >= 0.4:
                preds.append(row["diff_labels"])
                probs.append(row["diff_probs"])
            elif row["trie_probs"] >= 0.25:
                preds.append(row["trie_labels"])
                probs.append(row["trie_probs"])
            else:
                preds.append(row["bert_labels"])
                probs.append(row["bert_probs"]/3)

    return preds, probs

    


if __name__ == "__main__":

    # Document extraction
    scanner = DocScanner()
    os.makedirs(args.output,exist_ok=True)
    scanner.scan(args.input, PREPROCESS_RES)

    # Text detection model + OCR model config
    det_config = detection.Config("tool/config/detection/configs.yaml")
    os.environ['CUDA_VISIBLE_DEVICES'] = det_config.gpu_devices
    ocr_config = ocr.Config.load_config_from_file(OCR_CONFIG)
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
        num_boxes=TOP_K,
        save_csv=False)

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
        crop_region=True,
        save_csv=True)
    
    img = detection.draw_bbox(cv2.imread(PREPROCESS_RES)[:, :, ::-1], boxes_list)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(DETECTION_RES, img)

    # OCR
    df = pd.read_csv(DETECTION_CSV_RES)

    img_crop_names = df.box_names.tolist()
    # img_crop_names.sort(key=ocr.natural_keys)
    crop_texts = []
    for i, img_crop in enumerate(img_crop_names):
        img_crop_path = os.path.join(DETECTION_FOLDER_RES, img_crop)
        img = Image.open(img_crop_path)
        text = ocr_model.predict(img)
        crop_texts.append(text)
    df["texts"] = crop_texts
    df.to_csv(DETECTION_CSV_RES, index=False)
    crop_texts = '||'.join(crop_texts)
    
    with open(OCR_RES, 'w+') as f:
        f.write(crop_texts)


    # Information Retrieval

    inputs = df.texts.tolist()

    ## Use BERT
    meta_data = torch.load(BERT_WEIGHT)
    cfg = meta_data["config"]
    model_state = meta_data["model_state_dict"]

    retr_model = retrieval.get_instance(cfg["model"]).cuda()
    retr_model.load_state_dict(model_state)

    dataset = retrieval.MCOCRDataset_from_list(
        inputs, pretrained_model=cfg["model"]["args"]["pretrained_model"], max_len=31,
        preproc=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, pin_memory=False
    )

    with torch.no_grad():
        preds, probs = retrieval.inference(model=retr_model, dataloader=dataloader, device=torch.device("cuda:0"))
    
    df["bert_labels"] = [IDX_TO_LABEL[x] for x in preds]
    df["bert_probs"] = probs

    ## HEURISTICS
    retr_df = pd.read_csv('./retrieval/heuristic/custom-dictionary.csv')
    retr_texts = {}
    for id, row in retr_df.iterrows():
        retr_texts[row.text.lower()] = row.lbl

    inference = retrieval.get_heuristic_retrieval('diff')
    preds, probs = inference(inputs,retr_texts)

    df["diff_labels"] = [IDX_TO_LABEL[x] for x in preds]
    df["diff_probs"] = probs

    inference = retrieval.get_heuristic_retrieval('trie')
    preds, probs = inference(inputs,retr_texts)

    df["trie_labels"] = [IDX_TO_LABEL[x] for x in preds]
    df["trie_probs"] = probs

    ## TIMESTAMPS
    timestamps = retrieval.regex_timestamp(inputs)

    df["timestamp"] = timestamps

    ## Merge results
    preds, probs = merge_result(df)
    df["labels"] = preds
    df["probs"] = probs
    df.to_csv(DETECTION_CSV_RES, index=False)

    # Visualize result
    img = cv2.imread(DETECTION_RES)
    visualize(img, df.boxes.tolist(), df.texts.tolist(), df.labels.tolist(), df.probs.tolist(), FINAL_RES)
    
