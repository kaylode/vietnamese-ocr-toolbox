import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("./retrieval/text_classification/libs/")

from customdatasets import MCOCRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.getter import get_instance

from string import punctuation
import re


def clean(s):
    res = re.sub(r"(\w)(\()(\w)", "\g<1> \g<2>\g<3>", s)
    res = re.sub(r"\b\d+\b", "", res)
    res = re.sub(r"(\w)([),.:;]+)(\w)", "\g<1>\g<2> \g<3>", res)
    res = re.sub(r"(\w)(\.\()(\w)", "\g<1>. (\g<3>", res)
    res = re.sub(r"\s+", " ", res)
    res = res.strip()
    return res


def stripclean(arr):
    res = [s.strip().strip(punctuation) for s in arr]
    return " ".join([i for i in res if i != ""])


def dummy(x):
    # stupid workaround to deep copy array cause i couldn't get it to work properly
    return [s for s in x]


class MCOCRDataset_from_list(MCOCRDataset):
    def __init__(self, ls, pretrained_model, max_len, clean=False):
        self.is_train = False
        self.max_len = max_len
        self.df = pd.DataFrame.from_dict({"text": ls, "lbl": len(ls) * [0],})
        if clean:
            self.df["text"] = self.df["text"].apply(clean).str.split().apply(stripclean)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


def inference(model, dataloader, device):
    model.eval()
    preds = []
    targets = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, x in pbar:
        input_ids = x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)
        outputs = model(input_ids, attention_mask)
        outputs = F.softmax(outputs, dim=1)
        preds += [(outputs.detach().cpu().numpy())]

    preds = np.concatenate(preds, axis=0)

    return np.argmax(preds, 1), np.max(preds, 1)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta_data_path = sys.argv[1]  #'model.pth'
    meta_data = torch.load(meta_data_path)
    cfg = meta_data["config"]
    model_state = meta_data["model_state_dict"]

    inputs = [
        "co.op mart",
        "Co.opMart HAU GIANG",
        "188 Hau Giang, P.6, Q.6, TpHCM",
        "Dat hang qua DT: 028.39.600.913",
        "Ngày: 21/05/2020",
        "20 : 42 : 52",
        "Tong so tien thanh toan:",
        "16,200.00",
    ]
    model = get_instance(cfg["model"]).to(device)
    model.load_state_dict(model_state)

    dataset = MCOCRDataset_from_list(
        inputs, pretrained_model=cfg["model"]["args"]["pretrained_model"], max_len=31,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, pin_memory=False
    )

    # Run
    lbl_dict = {0: "SELLER", 1: "ADDRESS", 2: "TIMESTAMP", 3: "TOTAL_COST"}
    with torch.no_grad():
        preds, probs = inference(model=model, dataloader=dataloader, device=device)

    res = list(zip(inputs, [lbl_dict[x] for x in preds], probs))
    print(res)
    # Clean up
    del model, model_state
