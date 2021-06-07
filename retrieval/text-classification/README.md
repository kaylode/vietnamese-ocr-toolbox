# commonlit-readability-prize  

Rate the complexity of literary passages for grades 3-12 classroom use

## Installation

Requires Python 3.6.x (have not tested with newer version yet!)

```
sudo apt-get install python3-venv  
python3 -m venv venv  
source venv/bin/activate  
pip3 install -r requirements.txt  
```

## Notebooks

GPU Training: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MJ5XTHEUQdWlwvJDOcR2gLxp1vTnJLIW?usp=sharing)  

[Inference notebook](https://www.kaggle.com/kvu207/crp-inference-notebook-single-model) (**Requires logging in**)

## Relevent blog posts (by other competitors)

[Best Single Model](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/236645)  
[My Experience So Far and Further Improvements](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/241029)  
[AutoNLP to the rescue](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/237795)  
[Don't Jump to conclusions: Crazy variance w/ random seeds](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/242411)  

## Public kernels

[Step 1: Create Folds](https://www.kaggle.com/abhishek/step-1-create-folds)  
[CommonLit Readability Prize - RoBERTa Torch|ITPT](https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-itpt)  
[RoBERTa Base Fine-Tuning with Better Training Strategies](https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-fit)  

## Results

### Single model

| Model | Settings | Val (CV if 5 folds) | LB |
| --- | --- | --- | --- |
| AutoNLP | fold 0 (old) | 0.5020 | 0.518 |
| AutoNLP | fold 1 (old) | 0.5139 | 0.503 |
| AutoNLP | fold 2 (old) | 0.5285 |  |
| AutoNLP | fold 3 (old) | 0.5271 |  |
| AutoNLP | fold 4 (old) | 0.5246 |  |
| AutoNLP | 5 folds avg (old)| 0.5192 | 0.482 |
| bert-base-uncased | fold 0 | 0.4963 | 0.530 |
| bert-base-uncased | fold 1 | 0.5227 | 0.550 |
| bert-base-uncased | fold 2 | 0.5251 |  |
| bert-base-uncased | fold 3 | 0.5124 |  |
| bert-base-uncased | fold 4 | 0.5382 |  |
| bert-base-uncased | 5 folds avg | 0.5189 | 0.524 |

