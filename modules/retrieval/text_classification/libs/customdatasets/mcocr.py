import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MCOCRDataset(Dataset):
    def __init__(self, pretrained_model, csv_path, is_train, max_len):
        self.csv_path = csv_path
        self.is_train = is_train
        self.max_len = max_len
        self.df = pd.read_csv(self.csv_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        text = str(row.text).lower()
        target = row.lbl if self.is_train else None
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors="pt",
        )

        if self.is_train:
            target = torch.tensor(target, dtype=torch.long)
            return {
                "text": text,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "target": target,
            }
        else:
            return {
                "text": text,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }


class MCOCRDataset_from_list(MCOCRDataset):
    def __init__(self, ls, pretrained_model, max_len):
        # super(MCOCRDataset_from_list, self).__init__(
        #     pretrained_model=pretrained_model, csv_path=None, max_len=max_len
        # )
        # self.csv_path = csv_path
        # self.is_train = is_train
        # self.max_len = max_len
        # self.df = pd.read_csv(self.csv_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.is_train = False
        self.max_len = max_len
        self.df = pd.DataFrame.from_dict({"text": ls, "lbl": len(ls) * [0],})
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


if __name__ == "__main__":
    dataset = MCOCRDataset(
        pretrained_model="vinai/phobert-base",
        csv_path="data/splitted_train_val/train.csv",
        is_train=True,
        max_len=31,
    )
    dataset[0]
