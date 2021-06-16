import os
import torch
from .map import mAPScores
from .metrics import *
from .cal_recall.script import  cal_recall_precison_f1


def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text > 0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def cal_kernel_score(kernel, gt_kernel, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks.float()).data.cpu().numpy()
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel > 0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel
    

def get_metric(config):
    metric = mAPScores(
        ann_file= os.path.join('./data', config.project_name, config.val_anns),
        img_dir=os.path.join('./data', config.project_name, config.val_imgs)
    )

    return metric
