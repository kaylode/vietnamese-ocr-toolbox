import torch
import torchvision.transforms as tf
import os
import cv2
import time
from models import get_model
from config import Config
from post_processing import decode_clip
import argparse

def expand_box(img, boxes):
    h,w,c = img.shape
    new_boxes = np.array(boxes)
   
    for i, box in enumerate(new_boxes):
        x,y,w,h = box
        if w>h:
            new_boxes[i, 0] -= new_boxes[:, 3]
            new_boxes[i, 2] += (2*new_boxes[:, 3])
        elif w<h:
            new_boxes[i, 1] -= new_boxes[:, 3]
            new_boxes[i, 3] += (2*new_boxes[:, 2])

    return new_boxes

def sort_box(boxes):
    sorted_boxes = sorted(boxes , key=lambda k: [k[0][1], k[0][0]])
    return sorted_boxes

def crop_box(img, boxes, image_name, out_folder):
    h,w,c = img.shape
    sorted_boxes = sort_box(boxes)
    # new_boxes = expand_box(img, sorted_boxes)
    for i, box in enumerate(sorted_boxes):
        box_name = os.path.join(out_folder, f"{i}.jpg")
        
        (x1,y1),(x2,y2),(x3,y3),(x4,y4) = box
        x1,y1,x2,y2,x3,y3,x4,y4 = int(x1),int(y1),int(x2),int(y2),int(x3),int(y3),int(x4),int(y4)
        min_x = max(0, min(x1,x2,x3,x4))
        min_y = max(0, min(y1,y2,y3,y4))
        max_x = min(w, max(x1,x2,x3,x4))
        max_y = min(h, max(y1,y2,y3,y4))

        cropped = img[min_y:max_y, min_x:max_x, :]

        try:
            cv2.imwrite(box_name, cropped)
        except:
            print(box_name, " is missing")



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

    def predict(self, img_path: str, output_dir:str =None, short_size: int = 736, crop_region=False):
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
            crop_box(ori_img, boxes_list, image_name, output_dir)
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