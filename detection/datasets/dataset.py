import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .data_utils import image_label
from ..utils import order_points_clockwise
from pycocotools.coco import COCO

class ImageDataset(Dataset):
    def __init__(self, images_dir: str, ann_path: str, input_size: int, img_channel: int, shrink_ratio: float, transform=None,
                 target_transform=None, train=True):

        self.root_dir = images_dir
        self.data_list = self.load_data(ann_path)
        self.train = train
        self.input_size = input_size
        self.img_channel = img_channel
        self.transform = transform
        self.target_transform = target_transform
        self.shrink_ratio = shrink_ratio

    def __getitem__(self, index):
        img_path, text_polys, text_tags = self.data_list[index]
        im = cv2.imread(img_path, 1 if self.img_channel == 3 else 0)
        if self.img_channel == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        img, score_map, training_mask = image_label(im, text_polys, text_tags, self.input_size,
                                                    self.shrink_ratio, degrees=90, train=self.train)
        # img = draw_bbox(img,text_polys)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            score_map = self.target_transform(score_map)
            training_mask = self.target_transform(training_mask)
        return img, score_map, training_mask

    def load_data(self, ann_path: str) -> list:
        t_data_list = []
        
        self.coco = COCO(ann_path)
        self.image_ids = self.coco.getImgIds()

        for img_id in self.image_ids:
            img_path = self._get_image(img_id)
            bboxs, text_tags = self._get_annotation(img_id)
            if len(bboxs) > 0:
                t_data_list.append((img_path, bboxs, text_tags))
            else:
                print('there is no suit bbox in {}'.format(label_path))
        return t_data_list

    def _get_image(self, index: int) -> str:
        image_info = self.coco.loadImgs(index)[0]
        path = os.path.join(self.root_dir, image_info['file_name'])
        return path

    def _get_annotation(self, index: int) -> tuple:
        def convert_bbox_to_polygon(bbox):
            # x,y,w,h
            x,y,w,h = bbox
            p1 = [x,y]
            p2 = [x+w, y]
            p3 = [x+w, y+h]
            p4 = [x, y+h]
            return [p1,p2,p3,p4]

        boxes = []
        text_tags = []
        
        annotations_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)

        for idx, a in enumerate(coco_annotations):
            box = convert_bbox_to_polygon(a['bbox'])
            boxes.append(box)
            
            text_tags.append(a['tag'])
        return np.array(boxes, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def __len__(self):
        return len(self.data_list)
