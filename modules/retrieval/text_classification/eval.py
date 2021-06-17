import sys

sys.path.append("./libs/")
import numpy as np
import torch
import torch.nn as nn
from customdatasets import MCOCRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.getter import get_instance


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
    meta_data_path = sys.argv[1]  #'model.pth'
    meta_data = torch.load(meta_data_path)
    cfg = meta_data["config"]
    model_state = meta_data["model_state_dict"]

    model = get_instance(cfg["model"]).to(device)
    model.load_state_dict(model_state)

    dataset = MCOCRDataset(
        pretrained_model=cfg["model"]["args"]["pretrained_model"],
        csv_path=cfg["dataset"]["val"]["args"]["csv_path"],
        is_train=True,
        max_len=31,
    )
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, pin_memory=False
    )

    # Run
    with torch.no_grad():
        preds, targets = inference(model=model, dataloader=dataloader, device=device)

    metric = {mcfg["name"]: get_instance(mcfg) for mcfg in cfg["metric"]}
    for m in metric.values():
        preds = torch.Tensor(preds)
        targets = torch.Tensor(targets)
        value = m.calculate(preds, targets)
        m.update(value)
        m.summary()

    # Clean up
    del model, model_state
