import os
from ocr.tool.config import Cfg
from ocr.model.trainer import Trainer
from ocr.tool.predictor import Predictor
from tool.config import Config


if __name__ == '__main__':
    config = Config("tool/config/ocr/configs.yaml")

    model_config = Cfg.load_config_from_name(config.model_name)

    dataset_params = {
        "name": config.project_name,
        "data_root": config.data_root,
        "train_annotation": config.train_annotation,
        "valid_annotation": config.valid_annotation,
    }

    params = {
        "print_every": config.print_every,
        "valid_every": config.valid_every,
        "iters": config.iters,
        "export": config.export
    }

    model_config["trainer"].update(params)
    model_config["dataset"].update(dataset_params)
    model_config["device"] = config.gpu_devices

    trainer = Trainer(model_config, pretrained=True)
    trainer.train()
