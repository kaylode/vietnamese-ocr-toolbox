import json
from glob import glob
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

mcocr_root = Path("./mcocr_public/mcocr_train_data/")
mcocr_ims_dir = mcocr_root / "train_images"
mcocr_csv_path = mcocr_root / "mcocr_train_df.csv"

ocr_savedir = Path("./ocr_data/img")
ocr_savedir.mkdir(parents=True, exist_ok=True)
ocr_txt = Path("./ocr_data") / "data.txt"

if __name__ == "__main__":
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

