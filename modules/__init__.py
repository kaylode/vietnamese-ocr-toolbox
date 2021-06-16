import os
import cv2
import shutil
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from .preprocess import DocScanner
import modules.detection as detection
import modules.ocr as ocr
import modules.retrieval as retrieval
import modules.correction as correction
from tool.config import Config 
from tool.utils import download_pretrained_weights


CACHE_DIR = '.cache'

class Preprocess:
    def __init__(
        self, 
        find_best_rotation=True,
        det_model=None,
        ocr_model=None):
        
        self.find_best_rotation = find_best_rotation

        if self.find_best_rotation:
            self.crop_path = os.path.join(CACHE_DIR, 'crops')
            if os.path.exists(self.crop_path):
                shutil.rmtree(self.crop_path)
                os.mkdir(self.crop_path)
            self.det_model = det_model if det_model is not None else Detection()
            self.ocr_model = ocr_model if ocr_model is not None else OCR()
        self.scanner = DocScanner()

    def __call__(self, image, return_score=False):
        

        output = self.scanner.scan(image)
        
        if self.find_best_rotation:

            _ = self.det_model(
                output,
                crop_region=True,
                return_result=False,
                output_path=CACHE_DIR)

            orientation_scores = np.array([0.,0.,0.,0.])
            num_crops = len(os.listdir(self.crop_path))
            for i in range(num_crops):
                single_crop_path = os.path.join(self.crop_path, f'{i}.jpg')
                if not os.path.isfile(single_crop_path):
                    continue
                img = cv2.imread(single_crop_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orientation_scores += ocr.find_rotation_score(img, self.ocr_model)
            best_orient = np.argmax(orientation_scores)
            print(f"Rotate image by {best_orient*90} degrees")

            # Rotate the original image
            output = ocr.rotate_img(output, best_orient)
        
        if return_score:
            return output, orientation_scores
        else:
            return output

class Detection:
    def __init__(self, config_path=None, weight_path=None, model_name=None):
        if config_path is None:
            config_path = 'tool/config/detection/configs.yaml'
        config = Config(config_path)
        self.model_name = model_name
        if weight_path is None:
            if self.model_name is None:
                self.model_name = "pan_resnet18_default"
            tmp_path = os.path.join(CACHE_DIR, f'{self.model_name}.pth')
            download_pretrained_weights(self.model_name, cached=tmp_path)
            weight_path = tmp_path
        self.model = detection.PAN(config, model_path=weight_path)
        
    def __call__(
        self, 
        image,
        crop_region=False,
        return_result=False,
        output_path=None):
        
        """
        Input: path to image
        Output: boxes (coordinates of 4 points)
        """

        if output_path is None:
            assert crop_region, "Please specify output_path"
        else:
            output_path = os.path.join(output_path, 'crops')
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
                os.mkdir(output_path)

            
        # Detect and OCR for final result
        _, boxes_list, _ = self.model.predict(
            image, 
            output_path, 
            crop_region=crop_region)

        if return_result:
            img = detection.draw_bbox(image, boxes_list)
        
        if return_result:
            return boxes_list, img
        else:
            return boxes_list

class OCR:
    def __init__(self, config_path=None, weight_path=None, model_name=None):
        if config_path is None:
            config_path = 'tool/config/ocr/configs.yaml'
        config = Config(config_path)
        ocr_config = ocr.Config.load_config_from_name(config.model_name)
        ocr_config['cnn']['pretrained']=False
        ocr_config['device'] = 'cuda:0'
        ocr_config['predictor']['beamsearch']=False

        self.model_name = model_name
        if weight_path is None:
            if self.model_name is None:
                self.model_name = "transformerocr_default_vgg"
            tmp_path = os.path.join(CACHE_DIR, f'{self.model_name}.pth')
            download_pretrained_weights(self.model_name, cached=tmp_path)
            weight_path = tmp_path
        ocr_config['weights'] = weight_path
        self.model = ocr.Predictor(ocr_config)

    def __call__(self, img, return_prob=False):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.model.predict(img, return_prob)

    def predict_folder(self, img_paths, return_probs=False):
        texts = []
        if return_probs:
            probs = []
        for i, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            if return_probs:
                text, prob = self(img, True)
                texts.append(text)
                probs.append(prob)
            else:
                text = self(img, False)
                texts.append(text)

        if return_probs:
            return texts, probs
        else:
            return texts

class Retrieval:
    def __init__(self, class_mapping, dictionary=None, mode="all", bert_weight=None):
        assert mode in ["all", "bert", "trie", "ed"], "Mode is not supported"
        self.mode = mode

        self.dictionary = dictionary
        self.class_mapping = class_mapping
        self.idx_mapping = {v:k for k,v in class_mapping.items()}

        if self.mode == 'bert':
            self.use_bert = True
        if self.mode == 'trie':
            self.use_trie = True
        if self.mode == 'ed':
            self.use_ed = True
        if self.mode == 'all':
            self.use_bert = True
            self.use_trie = True
            self.use_ed = True

        if self.use_bert:
            self.bert = retrieval.PhoBERT(self.idx_mapping, bert_weight)
        if self.use_ed:
            self.ed = retrieval.get_heuristic_retrieval('diff')
        if self.use_trie:
            self.trie = retrieval.get_heuristic_retrieval('trie')

        if self.use_ed or self.use_trie:
            if self.dictionary is None:
                self.dictionary = {}
                df = pd.read_csv('./modules/retrieval/heuristic/custom-dictionary.csv')
                for id, row in df.iterrows():
                    self.dictionary[row.text.lower()] = row.lbl

    def ensemble(self, df):
        preds = []
        probs = []

        for id, row in df.iterrows():
            if row["timestamp"] == 1:
                preds.append("TIMESTAMP")
                probs.append(5.0)
            elif row["bert_labels"] == row["diff_labels"]:
                preds.append(row["bert_labels"])
                probs.append(row["bert_probs"] + row["diff_probs"])
            elif row["bert_labels"] == row["trie_labels"]:
                preds.append(row["bert_labels"])
                probs.append(row["bert_probs"] + row["trie_probs"])
            elif row["trie_labels"] == row["diff_labels"]:
                preds.append(row["trie_labels"])
                probs.append(row["trie_probs"] + row["diff_probs"])
            else:
                if row["diff_probs"] >= 0.4:
                    preds.append(row["diff_labels"])
                    probs.append(row["diff_probs"])
                elif row["trie_probs"] >= 0.25:
                    preds.append(row["trie_labels"])
                    probs.append(row["trie_probs"])
                else:
                    preds.append(row["bert_labels"])
                    probs.append(row["bert_probs"]/3)

        return preds, probs

    def __call__(self, query_texts):
        df = pd.DataFrame()
        if self.use_bert:
            preds, probs = self.bert(query_texts)
            df["bert_labels"] = preds
            df["bert_probs"] = probs
        if self.use_ed:
            preds, probs = self.ed(query_texts, self.dictionary)
            df["diff_labels"] = [self.idx_mapping[x] for x in preds]
            df["diff_probs"] = probs
        if self.use_trie:
            preds, probs = self.trie(query_texts, self.dictionary)
            df["trie_labels"] = [self.idx_mapping[x] for x in preds]
            df["trie_probs"] = probs

        timestamps = retrieval.regex_timestamp(query_texts)
        df["timestamp"] = timestamps
        preds, probs = self.ensemble(df)
        return preds, probs

        
class Correction:
    def __init__(self, dictionary=None, mode="ed"):
        assert mode in ["trie", "ed"], "Mode is not supported"
        self.mode = mode
        self.dictionary = dictionary

        self.use_trie = False
        self.use_ed = False

        if self.mode == 'trie':
            self.use_trie = True
        if self.mode == 'ed':
            self.use_ed = True
        
        if self.use_ed:
            self.ed = correction.get_heuristic_correction('diff')
        if self.use_trie:
            self.trie = correction.get_heuristic_correction('trie')
        
        if self.use_ed or self.use_trie:
            if self.dictionary is None:
                self.dictionary = {}
                df = pd.read_csv('./modules/retrieval/heuristic/custom-dictionary.csv')
                for id, row in df.iterrows():
                    self.dictionary[row.text.lower()] = row.lbl

    def __call__(self, query_texts, return_score=False):
        if self.use_ed:
            preds, score = self.ed(query_texts, self.dictionary)
            
        if self.use_trie:
            preds, score = self.trie(query_texts, self.dictionary)
        
        if return_score:
            return preds, score
        else:
            return preds