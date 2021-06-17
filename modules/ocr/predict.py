import os
import re
from PIL import Image
import cv2
import numpy as np
from .tool.predictor import Predictor
from .tool.config import Cfg
import argparse

def find_rotation_score(img, detector):
    scores = []
    t, score = detector(img, return_prob=True)
    scores.append(score)
    new_img = img.copy()
    for i in range(3):
        new_img = cv2.rotate(new_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        t, score = detector(new_img, return_prob=True)
        scores.append(score)
    return np.array(scores)

def rotate_img(img, orient):
    new_img = img.copy()
    for i in range(orient):
        new_img = cv2.rotate(new_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return new_img
    

    
