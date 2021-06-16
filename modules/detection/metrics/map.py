"""
COCO Mean Average Precision Evaluation

True Positive (TP): Predicted as positive as was correct
False Positive (FP): Predicted as positive but was incorrect
False Negative (FN): Failed to predict an object that was there

if IOU prediction >= IOU threshold, prediction is TP
if 0 < IOU prediction < IOU threshold, prediction is FP

Precision measures how accurate your predictions are. Precision = TP/(TP+FP)
Recall measures how well you find all the positives. Recal = TP/(TP+FN)

Average Precision (AP) is finding the area under the precision-recall curve.
Mean Average  Precision (MAP) is AP averaged over all categories.

AP@[.5:.95] corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05
AP@.75 means the AP with IoU=0.75

*Under the COCO context, there is no difference between AP and mAP

"""

import os
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def _eval(coco_gt, image_ids, pred_json_path, **kwargs):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.params.iouThrs = np.array([0.5])
    # Some params for COCO eval
    #imgIds = []
    #catIds = []
    #iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    #recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    #maxDets = [1, 10, 100]
    #areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    #areaRngLbl = ['all', 'small', 'medium', 'large']
    #useCats = 1

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return stats

class mAPScores():
    def __init__(self, ann_file, img_dir):
        self.coco_gt = COCO(ann_file)
        self.img_dir = img_dir
        self.filepath = f'results/bbox_results.json'
        self.image_ids = self.coco_gt.getImgIds()

        self.reset()

        if not os.path.exists('results'):
            os.mkdir('results')

    def _get_image(self, index: int) -> str:
        image_info = self.coco_gt.loadImgs(index)[0]
        path = os.path.join(self.img_dir, image_info['file_name'])
        return path

    def reset(self):
        self.model = None

    def update(self, model):
        self.model = model
       

    def compute(self):
        results = []
        with torch.no_grad():

            with tqdm(total=len(self.image_ids)) as pbar:
                for img_id in self.image_ids:
                    
                    img_path = self._get_image(img_id)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    _, boxes_list, _ = self.model.predict(img)

                    boxes = []
                    labels = []
                    scores = []
                    for i, box in enumerate(boxes_list):
                        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
                        x1,y1,x2,y2,x3,y3,x4,y4 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)
                        min_x = min(x1,x2,x3,x4)
                        min_y = min(y1,y2,y3,y4)
                        max_x = max(x1,x2,x3,x4)
                        max_y = max(y1,y2,y3,y4)

                        box = [min_x, min_y, max_x-min_x, max_y-min_y]
   
                        image_result = {
                            'image_id': img_id,
                            'category_id': 1,
                            'score': 1.0,
                            'bbox': box,
                        }

                        results.append(image_result)

                    pbar.update(1)
                    
        if not len(results):
            return False

        # write output
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        json.dump(results, open(self.filepath, 'w'), indent=4)
        return True

    def value(self):
        result = self.compute()
        if result:
            stats = _eval(self.coco_gt, self.image_ids, self.filepath)
            return {
                "MAP" : np.round(float(stats[0]),4),
                "MAPsmall" : np.round(float(stats[3]),4),
                "MAPmedium" : np.round(float(stats[4]),4),
                "MAPlarge" : np.round(float(stats[5]),4),}
        else:
            return {
                "MAP" : 0.0,
                "MAPsmall" : 0.0,
                "MAPmedium" : 0.0,
                "MAPlarge" : 0.0,}
