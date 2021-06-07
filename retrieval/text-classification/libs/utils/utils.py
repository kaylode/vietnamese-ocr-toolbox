import numpy as np
import torch
import torchvision
import os


def load_file_from_folder(folder_path):
    assert folder_path != None, "Folder Path is empty!"
    return os.listdir(folder_path)


def execute_filename(filename):
    filename = filename.split("-")[1]
    return filename


def rescale(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))


def vprint(obj, vb):
    if vb:
        print(obj)
    return


class NormMaxMin:
    def __call__(self, x):
        return (x.float() - torch.min(x)) / (torch.max(x) - torch.min(x))
