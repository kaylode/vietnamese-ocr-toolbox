import torch
import torchvision.transforms as tf
import os
import cv2
import time
import numpy as np
import pandas as pd
from detection.models import get_model
from tool.config import Config
from detection.utils.util import order_points_clockwise
from detection.post_processing import decode_clip
import argparse

def sort_box(boxes):
    sorted_boxes = []
    for box in boxes:
        sorted_boxes.append(order_points_clockwise(box))
    mid_points = [line_intersection((box[0],box[2]), (box[1], box[3])) for box in sorted_boxes]
    sorted_indices = np.argsort(mid_points, axis=0)
    sorted_boxes = sorted(sorted_boxes , key=lambda sorted_indices: [sorted_indices[0][1], sorted_indices[0][0]]) 
    return sorted_boxes

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def crop_box(img, boxes, image_name, out_folder, num_boxes=0, save_csv=True):
    h,w,c = img.shape
    sorted_boxes = sort_box(boxes)

    if save_csv:
        boxes = []
        box_names = []

    for i, box in enumerate(sorted_boxes):
        box_name = os.path.join(out_folder, f"{i}.jpg")
        
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
        x1,y1,x2,y2,x3,y3,x4,y4 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)
        x1 = max(0, x1)
        x2 = max(0, x2)
        x3 = max(0, x3)
        x4 = max(0, x4)
        y1 = max(0, y1)
        y2 = max(0, y2)
        y3 = max(0, y3)
        y4 = max(0, y4)
        min_x = max(0, min(x1,x2,x3,x4))
        min_y = max(0, min(y1,y2,y3,y4))
        max_x = min(w, max(x1,x2,x3,x4))
        max_y = min(h, max(y1,y2,y3,y4))
        
        if num_boxes==0:
            tw = int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
            th = int(np.sqrt((x1-x4)**2 + (y1-y4)**2))
            pt1 = np.float32([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])
            pt2 = np.float32([[0, 0],
                              [tw - 1, 0],
                              [tw - 1, th - 1],
                              [0, th - 1]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            cropped = cv2.warpPerspective(img, matrix, (tw, th)) 
            box_names.append(box_name)
            boxes.append([x1,y1,x2,y2,x3,y3,x4,y4])
        else:
            cropped = img[min_y:max_y, min_x:max_x, :]

        try:
            cv2.imwrite(box_name, cropped)
        except:
            print(box_name, " is missing")
        
        if num_boxes>0 and i == num_boxes-1:
            break

    if save_csv:
        data = {
            "box_names": box_names,
            "boxes": boxes
        }
        out_dir = os.path.dirname(out_folder)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(out_dir, "box_info.csv"), index=False)



class PAN:
    def __init__(self, config, model_path=None, state_dict=None):
        
        self.device = torch.device("cuda")
        self.net = get_model(config.model)
        if model_path is not None:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['state_dict'])
        elif state_dict is not None:
            self.net.load_state_dict(state_dict)
            
        self.net.to(self.device)
        self.net.eval()

    def predict(self, 
            img_path: str, 
            output_dir:str =None, 
            short_size: int = 736, 
            crop_region: bool =False, 
            num_boxes: int =0, 
            save_csv: bool = True):

        img = cv2.imread(img_path)
        ori_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = ori_img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(ori_img, None, fx=scale, fy=scale)

        tensor = tf.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            torch.cuda.synchronize(self.device)

            preds, boxes_list = decode_clip(preds)
            scale = (preds.shape[1] / w, preds.shape[0] / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
            t = time.time() - start

        image_name = os.path.basename(img_path)
        if crop_region:
            os.makedirs(output_dir, exist_ok=True)
            crop_box(ori_img, boxes_list, image_name, output_dir, num_boxes=num_boxes, save_csv=save_csv)
        return preds, boxes_list, t


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox

    parser = argparse.ArgumentParser("Inference PAN")
    parser.add_argument('--input', '-i', type=str, help='Path to input image')
    parser.add_argument('--output', '-o', type=str, help='Path to save output image')
    parser.add_argument('--weight', '-w', type=str, help='Path to trained model')
    args = parser.parse_args()

    config = Config(os.path.join('config','configs.yaml'))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    

    model = PAN(config, model_path=args.weight)
    preds, boxes_list, t = model.predict(args.input, args.output, crop_region=True)
    
    show_img(preds)
    img = draw_bbox(cv2.imread(args.input)[:, :, ::-1], boxes_list)
    show_img(img, color=True)
    plt.axis('off')
    
    out_dir = os.path.dirname(args.output)
    outpath = os.path.join(out_dir, "detected.jpg")
    plt.savefig(outpath,bbox_inches='tight')