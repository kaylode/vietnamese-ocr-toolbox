import json
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm
import random
import json
import os
from PIL import Image


def extract_mcocr(data_dir = "./data/mcocr_public/mcocr_train_data/", out_dir="./data/ocr_data/"):
    
    mcocr_root = Path(data_dir)
    mcocr_ims_dir = mcocr_root / "train_images"
    mcocr_csv_path = mcocr_root / "mcocr_train_df.csv"

    ocr_savedir = Path(f"{out_dir}/img")
    ocr_savedir.mkdir(parents=True, exist_ok=True)
    ocr_txt = Path(f"{out_dir}") / "data.txt"
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


def split_mcocr(data_dir = "./data/ocr_data"):
    
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

def coco_convert(data_dir, out_path):
    paths = os.listdir(data_dir)
    paths = [i for i in paths if i.endswith(".txt")]
    my_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    cat_dict = {
        "id": 1, 
        "name": "Text", 
        "supercategory": "Text",
        }

    my_dict['categories'].append(cat_dict)

    ann_id = 0
    
    for image_id, image_ann_path in tqdm(enumerate(paths)):
        img_name = image_ann_path[:-3] + "jpg"
        ann_path = os.path.join(data_dir, image_ann_path)
        img_path = os.path.join(data_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(img_path, " is missing")
            continue
        image_w = img.width
        image_h = img.height
        
        image_dict = {
            "id": image_id, 
            "width": image_w, 
            "height": image_h, 
            "file_name": img_name
        }

        
        try:
            with open(ann_path, 'r', encoding="utf-8", newline="") as f:
                data = f.read()

                for row in data.splitlines():
                    ann = row.split(',')
                    x1,y1,x2,y2,x3,y3,x4,y4 = ann[:8]
                    text = ann[8:]

                    box_w = int(x2) - int(x1)
                    box_h = int(y3) - int(y1)

                    ann_dict = {
                        "id": ann_id, 
                        "image_id": image_id, 
                        "category_id": 1, 
                        "area": box_w*box_h, 
                        "bbox": [int(x1),int(y1),box_w, box_h], 
                        "iscrowd": 0,
                        "tag":True
                    }

                    my_dict["annotations"].append(ann_dict)
                    ann_id += 1
            my_dict["images"].append(image_dict)
        except UnicodeDecodeError:
            print(f"{ann_path} can't be read")
            continue
        except ValueError:
            print(f"{ann_path} wrong format")
            continue
        
    with open(out_path, 'w') as outfile:
        json.dump(my_dict, outfile)


def convert_sroie19_to_coco(
    data_dir='./data/SROIE2019',
    out_dir='./data/sroie19'):

    import shutil

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'images', 'val'), exist_ok=True)
    
    train_dir = os.path.join(data_dir, "0325updated.task1train(626p)")
    val_dir = os.path.join(data_dir, "task1_2_test(361p)")
    val_txtdir = os.path.join(data_dir, "text.task1_2-test（361p)")

    for i in os.listdir(val_txtdir):
        src_file = os.path.join(val_txtdir, i)
        dst_file = os.path.join(val_dir, i)
        shutil.move(src_file, dst_file)

    train_out = os.path.join(out_dir, 'annotations', 'train.json')
    val_out = os.path.join(out_dir, 'annotations', 'val.json')

    coco_convert(train_dir, train_out)
    coco_convert(val_dir, val_out)

    train_img_dir=os.path.join(out_dir, 'images', 'train')
    val_img_dir=os.path.join(out_dir, 'images', 'val')
    for i in os.listdir(val_dir):
        if i.endswith('.jpg'):
            src_file = os.path.join(val_dir, i)
            dst_file = os.path.join(val_img_dir, i)
            shutil.move(src_file, dst_file)

    for i in os.listdir(train_dir):
        if i.endswith('.jpg'):
            src_file = os.path.join(train_dir, i)
            dst_file = os.path.join(train_img_dir, i)
            shutil.move(src_file, dst_file)