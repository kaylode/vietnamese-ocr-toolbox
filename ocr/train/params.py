dataset_params = {
    "name": "mcocr",
    "data_root": "../../../inputs/ocr_data/",
    "train_annotation": "train_annotation.txt",
    "valid_annotation": "val_annotation.txt",
}

params = {
    "print_every": 100,
    "valid_every": 5 * 100,
    "iters": 20000,
    "checkpoint": "./checkpoint/transformerocr_checkpoint.pth",
    "export": "./weights/transformerocr.pth",
    "iters": 10000,
}

