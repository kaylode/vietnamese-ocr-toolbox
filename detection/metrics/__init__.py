import os
from .map import mAPScores

def get_metric(config):
    metric = mAPScores(
        ann_file= os.path.join('../data', config.project_name, config.val_anns),
        img_dir=os.path.join('../data', config.project_name, config.val_imgs)
    )

    return metric
