import torch
import torchvision.transforms as tf
import os
import cv2
import time
from models import get_model
from config import Config
from post_processing import decode_clip
import argparse

parser = argparse.ArgumentParser("Inference PAN")
parser.add_argument('--input', '-i', type=str, help='Path to input image')
parser.add_argument('--output', '-o', type=str, help='Path to save output image')
parser.add_argument('--weight', '-w', type=str, help='Path to trained model')
args = parser.parse_args()
    

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
    sorted_boxes = sorted(boxes , key=lambda k: [k[1], k[0]])
    return sorted_boxes

def crop_box(img, boxes, image_name, out_folder):

    sorted_boxes = sort_box(boxes)
    new_boxes = expand_box(img, sorted_boxes)
    for i, box in enumerate(new_boxes):
        box_name = os.path.join(out_folder, image_name[:-4] +f"_{i}.jpg")
        x,y,w,h = box
        x,y,w,h = int(x), int(y), int(w), int(h)
        
        cropped = img[max(0,y):y+h, max(0,x):x+w, :] * 255

        try:
            cv2.imwrite(box_name, cropped)
        except:
            print(box_name, " is missing")



class PAN:
    def __init__(self, config, model_path):
        
        self.device = torch.device("cuda")
        self.net = get_model(config.model)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.net.to(self.device)
        self.net.eval()

    def predict(self, img: str, short_size: int = 736):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]
        scale = short_size / min(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)

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
        return preds, boxes_list, t


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox

    config = Config(os.path.join('config','configs.yaml'))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    

    model = PAN(config, args.weight)
    preds, boxes_list, t = model.predict(args.input)
    show_img(preds)
    img = draw_bbox(cv2.imread(args.input)[:, :, ::-1], boxes_list)
    show_img(img, color=True)
    plt.axis('off')
    plt.savefig(args.output,bbox_inches='tight')