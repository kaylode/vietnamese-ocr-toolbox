import sys

sys.path.append("./libs/")

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from datasets import MCOCRDataset
from models import AutoModelForClassification
from tqdm import tqdm

from sklearn.metrics import mean_squared_error


def inference(model, dataloader, device):
    model.eval()
    preds = []
    targets = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, x in pbar:
        input_ids = x["input_ids"].to(device)
        attention_mask = x["attention_mask"].to(device)
        target = x["target"].tolist()
        outputs = model(input_ids, attention_mask)
        preds += [outputs.detach().cpu().numpy()]
        targets.extend(target)

    preds = np.concatenate(preds, axis=0)
    return preds, targets


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pretrained = str(sys.argv[1])  #'bert-base-uncased'

    model = AutoModelForClassification(pretrained_model=pretrained)
    model.to(device)
    model_state_path = str(sys.argv[2])  #'model.pth'
    model_state = torch.load(model_state_path)["model_state_dict"]
    model.load_state_dict(model_state)

    dataset = MCOCRDataset(
        pretrained_model=pretrained,
        csv_path=f"data/k-folds/val/val_fold_{sys.argv[3]}.csv",
        is_train=True,
        max_len=31,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, pin_memory=False
    )

    # Run
    with torch.no_grad():
        preds, targets = inference(model=model, dataloader=dataloader, device=device)

    print((mean_squared_error(targets, preds)) ** 0.5)

    # Clean up
    del model, model_state

