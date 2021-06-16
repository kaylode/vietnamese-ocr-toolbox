import random
from pathlib import Path

from pandas.core.reshape.melt import wide_to_long

root = Path("./ocr_data")
raw_file_path = root / "data.txt"
train_file_path = root / "train_annotation.txt"
val_file_path = root / "val_annotation.txt"

val_ratio = 0.1


def writefile(data, filename):
    with open(filename, "w") as f:
        f.writelines(data)


if __name__ == "__main__":
    data = []
    with open(raw_file_path, "r") as f:
        data = f.readlines()
    random.shuffle(data)
    val_len = int(val_ratio * len(data))
    train_data = data[:-val_len]
    val_data = data[-val_len:]
    writefile(train_data, train_file_path)
    writefile(val_data, val_file_path)
