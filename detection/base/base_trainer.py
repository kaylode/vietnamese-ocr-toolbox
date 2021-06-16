# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:50
# @Author  : zhoujun

import os
import shutil
import pathlib
from pprint import pformat
import torch
from torch import nn
from datetime import datetime
from ..utils import setup_logger


class BaseTrainer:
    def __init__(self, args, config, model, criterion, metric, weights_init):
        self.config=config
        self.save_dir = os.path.join(args.saved_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.best_acc=0
        self.best_map = 0
        self.global_step = 0
        self.start_epoch = 1
        self.config = config

        self.model = model
        self.metric = metric
        self.criterion = criterion
        # logger and tensorboard
        
        self.epochs = self.config.num_epochs
        self.display_interval = args.print_per_iter
        self.val_interval = args.val_interval

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(self.save_dir)

        self.logger = setup_logger(os.path.join(self.save_dir, 'train_log'))
        self.logger.info(pformat(self.config))

        # device
        torch.manual_seed(2000)  # 为CPU设置随机种子
        gpu_devices = config.gpu_devices.split(',')
        num_gpus = len(gpu_devices)
        if num_gpus > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.logger.info(
                'train with gpu {} and pytorch {}'.format(self.config.gpu_devices, torch.__version__))
            self.gpus = {i: item for i, item in enumerate(gpu_devices)}
            self.device = torch.device("cuda:0")
            torch.cuda.manual_seed(2000)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(2000)  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.logger.info('train with cpu and pytorch {}'.format(torch.__version__))
            self.device = torch.device("cpu")
        self.logger.info('device {}'.format(self.device))
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model': ''}

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr_policy["lr"],
            weight_decay=config.lr_policy["weight_decay"]
        )

        if args.resume:
            self._load_checkpoint(args.resume, resume=True)
        elif config.pretrained:
            self._load_checkpoint(config.pretrained, resume=False)
        else:
            if weights_init is not None:
                model.apply(weights_init)

        if args.freeze_backbone:
            print("freeze model's backbone")
            self.model.freeze_backbone()

        self.scheduler = torch.optim.lr_scheduler.StepLR(
                            self.optimizer, 
                            step_size=config.lr_scheduler['step_size'],
                            gamma=config.lr_scheduler['gamma'])

        # 单机多卡
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            try:
                self.epoch_result = self._train_epoch(epoch)
                self.scheduler.step()
                self._on_epoch_finish()
                if epoch % self.val_interval == 0:
                    self._val_epoch(epoch)
            except torch.cuda.CudaError:
                self._log_memory_usage()
        self.writer.close()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval(self):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_epoch_finish(self):
        raise NotImplementedError

    def _on_train_finish(self):
        raise NotImplementedError

    def _log_memory_usage(self):
        if not self.with_cuda:
            return

        template = """Memory Usage: \n{}"""
        usage = []
        for deviceID, device in self.gpus.items():
            deviceID = int(deviceID)
            allocated = torch.cuda.memory_allocated(deviceID) / (1024 * 1024)
            cached = torch.cuda.memory_cached(deviceID) / (1024 * 1024)

            usage.append('    CUDA: {}  Allocated: {} MB Cached: {} MB \n'.format(device, allocated, cached))

        content = ''.join(usage)
        content = template.format(content)

        self.logger.debug(content)

    def _save_checkpoint(self, epoch, file_name, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            # 'config': self.config,
            # 'metrics': self.metrics
        }
        
        torch.save(state, file_name)
        if save_best:
            self.logger.info("Saving current best: {}".format(file_name))
        else:
            self.logger.info("Saving checkpoint: {}".format(file_name))

    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch'] + 1
            # self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger.info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger.info("finetune from checkpoint {}".format(checkpoint_path))

