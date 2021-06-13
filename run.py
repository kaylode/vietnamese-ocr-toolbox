import os
import cv2
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from preprocess import DocScanner
import detection
import ocr
import retrieval
from tool.config import Config 



parser = argparse.ArgumentParser("Document Extraction")
parser.add_argument("--input", help="Path to single image to be scanned")
parser.add_argument("--output", default="./results", help="Path to output folder")
args = parser.parse_args()


class Pipeline:
    def __init__(self, args, config):
        self.input = args.input
        self.output = args.output
        self.load_config(config)
        self.make_cache_folder()

        pass

    def load_config(self, config):
        self.det_weight = config.det_weight
        self.ocr_weight = config.ocr_weight
        self.ocr_config = config.ocr_config
        
        if config.use_bert:
            self.bert_weight = config.bert_weight

    def make_cache_folder(self):
        self.cache_folder = os.path.join(args.output, 'cache')
        os.makedirs(self.cache,exist_ok=True)
        self.preprocess_cache = os.path.join(self.cache_folder, "preprocessed.jpg")
        self.detection_cache = os.path.join(self.cache_folder, "detected.jpg")
        self.csv_cache = os.path.join(self.cache_folder, 'box_info.csv')
        self.crop_cache = os.path.join(self.cache_folder, 'crops')
        self.final_output = os.path.join(self.output, 'final.jpg')



    def forward(self):
        # Document extraction
        scanner = DocScanner()
        scanner.scan(self.input, PREPROCESS_RES)

if __name__ == '__main__':


