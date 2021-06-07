# Author: Zylo117

import torch
import torchvision
import numpy as np
from torch import nn
from .modules import YoloLoss, Yolov4, non_max_suppression, Yolov5


def get_yolo(model_name):

    version_name = model_name.split('v')[1]
    net = YoloBackbone(
        version_name=version_name,
        num_classes=1)
  
    return net

class BaseBackbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseBackbone, self).__init__()
        pass
    def forward(self, batch):
        pass
    def detect(self, batch):
        pass

class YoloBackbone(BaseBackbone):
    def __init__(
        self, 
        device,
        version_name='5s',
        num_classes=80, 
        pretrained_backbone_path=None, 
        max_pre_nms=None,
        max_post_nms=None,
        **kwargs):

        super(YoloBackbone, self).__init__(**kwargs)

        if max_pre_nms is None:
            max_pre_nms = 30000
        self.max_pre_nms = max_pre_nms

        if max_post_nms is None:
            max_post_nms = 300
        self.max_post_nms = max_post_nms

        version = version_name[0]
        if version=='4':
            version_mode = version_name.split('-')[1]
            self.name = f'yolov4-{version_mode}'
            self.model = Yolov4(
                cfg=f'./models/yolo/configs/yolov4-{version_mode}.yaml', ch=3, nc=num_classes
            )
        elif version =='5':
            version_mode = version_name[-1]
            self.name = f'yolov5{version_mode}'
            self.model = Yolov5(
                cfg=f'./models/yolo/configs/yolov5{version_mode}.yaml', ch=3, nc=num_classes
            )
        

        if pretrained_backbone_path is not None:
            ckpt = torch.load(pretrained_backbone_path, map_location='cpu')  # load checkpoint
            try:
                self.model.load_state_dict(ckpt, strict=False) 
            except:
                pass

        self.model = nn.DataParallel(self.model).cuda()
        self.loss_fn = YoloLoss(
            num_classes=num_classes,
            model=self.model)

        self.num_classes = num_classes

    def forward(self, batch, device):
        inputs = batch["imgs"]
        targets = batch['yolo_targets']

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        if self.model.training:
            outputs = self.model(inputs)
        else:
            _ , outputs = self.model(inputs)

        loss, loss_items = self.loss_fn(outputs, targets)

        ret_loss_dict = {
            'T': loss,
            'IOU': loss_items[0],
            'OBJ': loss_items[1],
            'CLS': loss_items[2],
        }
        return ret_loss_dict

    def detect(self, batch, device):
        inputs = batch["imgs"]
        inputs = inputs.to(device)
        outputs, _ = self.model(inputs)
        outputs = non_max_suppression(
            outputs, 
            conf_thres=0.0001, 
            iou_thres=0.8, 
            max_nms=self.max_pre_nms,
            max_det=self.max_post_nms) #[bs, max_det, 6]
    
        out = []
        for i, output in enumerate(outputs):
            # [x1,y1,x2,y2, score, label]
            if output is not None and len(output) != 0:
                output = output.detach().cpu().numpy()
                boxes = output[:, :4]
                boxes[:,[0,2]] = boxes[:,[0,2]] 
                boxes[:,[1,3]] = boxes[:,[1,3]] 

                # Convert labels to COCO format
                labels = output[:, -1] + 1
                scores = output[:, -2]
          
            else:
                boxes = []
                labels = []
                scores = []
            if len(boxes) > 0:
                out.append({
                    'bboxes': boxes,
                    'classes': labels,
                    'scores': scores,
                })
            else:
                out.append({
                    'bboxes': np.array(()),
                    'classes': np.array(()),
                    'scores': np.array(()),
                })

        return out



    
