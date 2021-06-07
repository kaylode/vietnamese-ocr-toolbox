from torch.optim import SGD, Adam, RMSprop, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader

from losses import *
from customdatasets import *
from models import *
from metrics import *
from optimizers import *
from dataloaders import *
from schedulers import *
from externals import *


def get_instance(config, **kwargs):
    assert "name" in config
    config.setdefault("args", {})
    if config["args"] is None:
        config["args"] = {}
    return globals()[config["name"]](**config["args"], **kwargs)
