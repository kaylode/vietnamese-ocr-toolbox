import torch


def multi_class_prediction(output):
    return torch.argmax(output, dim=1)


def binary_prediction(output, thresh=0.0):
    return (output.squeeze(1) > thresh).long()
