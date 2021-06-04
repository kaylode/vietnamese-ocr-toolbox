from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
from params import *
from vietocr.tool.predictor import Predictor


config = Cfg.load_config_from_name("vgg_transformer")


config["trainer"].update(params)
config["dataset"].update(dataset_params)
config["device"] = "cuda:0"

trainer = Trainer(config, pretrained=True)
trainer.config.save("config.yml")
trainer.train()
