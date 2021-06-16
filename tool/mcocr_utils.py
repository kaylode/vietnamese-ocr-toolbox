import json
from glob import glob
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
import random


def extract_mcocr(data_dir = "./mcocr_public/mcocr_train_data/"):
    
    mcocr_root = Path(data_dir)
    mcocr_ims_dir = mcocr_root / "train_images"
    mcocr_csv_path = mcocr_root / "mcocr_train_df.csv"

    ocr_savedir = Path("./ocr_data/img")
    ocr_savedir.mkdir(parents=True, exist_ok=True)
    ocr_txt = Path("./ocr_data") / "data.txt"
    im_ls = list(mcocr_ims_dir.rglob("*.jpg"))
    df = pd.read_csv(mcocr_csv_path)
    count = 0
    with open(ocr_txt, "w") as f:
        for im in tqdm(im_ls):
            im_id = im.name
            img = cv2.imread(str(mcocr_ims_dir / im_id))
            query_df = df[df["img_id"] == im_id]
            clean_q = str(query_df["anno_polygons"].values[0]).replace("'", '"')
            clean_t = str(query_df["anno_texts"].values[0])
            text_ls = clean_t.split("|||")
            json_ls = json.loads(clean_q)
            for text, fjson in zip(text_ls, json_ls):
                fjson["bbox"] = list(map(int, fjson["bbox"]))
                x, y, w, h = fjson["bbox"]
                crop_img = img[y : y + h, x : x + w]
                if h > w:
                    crop_img = cv2.rotate(crop_img, cv2.cv2.ROTATE_90_CLOCKWISE)

                save_id = f"{count:05}.jpg"
                cv2.imwrite(f"{ocr_savedir/save_id}", crop_img)
                f.write(f"img/{save_id}\t{text}\n")
                count += 1


def split_mcocr(data_dir = "./ocr_data"):
    
    root = Path(data_dir)
    raw_file_path = root / "data.txt"
    train_file_path = root / "train_annotation.txt"
    val_file_path = root / "val_annotation.txt"

    val_ratio = 0.1


    def writefile(data, filename):
        with open(filename, "w") as f:
            f.writelines(data)


    data = []
    with open(raw_file_path, "r") as f:
        data = f.readlines()
    random.shuffle(data)
    val_len = int(val_ratio * len(data))
    train_data = data[:-val_len]
    val_data = data[-val_len:]
    writefile(train_data, train_file_path)
    writefile(val_data, val_file_path)
