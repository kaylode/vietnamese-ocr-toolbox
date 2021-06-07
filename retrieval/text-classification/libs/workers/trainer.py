import os
from datetime import datetime

import numpy as np
import torch
from loggers import TensorboardLogger
from torch import nn, optim
from torch.utils import data
from torchnet import meter
from tqdm import tqdm
from utils.device import detach, move_to
from torch.cuda.amp import GradScaler, autocast
from utils.utils import vprint


class Trainer:
    def __init__(self, device, config, model, criterion, optimizer, scheduler, metric):
        super(Trainer, self).__init__()

        self.config = config
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.scaler = GradScaler()
        self.iters = self.config["iters"]
        # Train ID
        self.train_id = str(self.config.get("id", "None"))
        # self.train_id += "-" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

        # Get arguments
        self.fp16 = self.config["fp16"]
        self.nepochs = self.config["trainer"]["nepochs"]
        self.log_step = self.config["trainer"]["log_step"]
        self.val_step = self.config["trainer"]["val_step"]
        self.debug = self.config["debug"]
        # self.tta_times = self.config["TTA_times"]
        self.eval_models = []
        self.verbose = self.config["verbose"]
        # Instantiate global variables
        self.max_grad_norm = 1.0
        self.best_loss = np.inf
        self.best_metric = {k: 0.0 for k in self.metric.keys()}
        self.val_loss = list()
        self.val_metric = {k: list() for k in self.metric.keys()}

        # Instantiate loggers
        self.save_dir = os.path.join("runs", self.train_id)
        self.tsboard = TensorboardLogger(path=self.save_dir)

    def save_checkpoint(self, epoch, val_loss, val_metric):

        data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        if val_loss < self.best_loss:
            vprint(
                f"Loss is improved from {self.best_loss: .6f} to {val_loss: .6f}. Saving weights...",
                self.verbose,
            )
            torch.save(data, os.path.join(self.save_dir, "best_loss.pth"))
            # Update best_loss
            self.best_loss = val_loss
        else:
            vprint(f"Loss is not improved from {self.best_loss:.6f}.", self.verbose)

        for k in self.metric.keys():
            if val_metric[k] > self.best_metric[k]:
                vprint(
                    f"{k} is improved from {self.best_metric[k]: .6f} to {val_metric[k]: .6f}. Saving weights...",
                    self.verbose,
                )
                torch.save(data, os.path.join(self.save_dir, f"best_metric_{k}.pth"))
                self.best_metric[k] = val_metric[k]
            else:
                vprint(
                    f"{k} is not improved from {self.best_metric[k]:.6f}.", self.verbose
                )

        # vprint('Saving current model...',self.verbose)
        # torch.save(data, os.path.join(self.save_dir, 'current.pth'))

    def train_epoch(self, epoch, dataloader):
        # 0: Record loss during training process
        running_loss = meter.AverageValueMeter()
        total_loss = meter.AverageValueMeter()
        for m in self.metric.values():
            m.reset()
        self.model.train()
        vprint("Training........", self.verbose)
        progress_bar = tqdm(dataloader) if self.verbose else dataloader
        for i, x in enumerate(progress_bar):
            if (i > self.iters) and (self.iters != -1):
                break
            # 1: Load img_inputs and labels
            input_ids = move_to(x["input_ids"], self.device)
            attention_mask = move_to(x["attention_mask"], self.device)
            target = move_to(x["target"], self.device)
            # 2: Clear gradients from previous iteration
            self.optimizer.zero_grad()
            with autocast(enabled=self.fp16):
                # 3: Get network outputs
                outs = self.model(input_ids, attention_mask)

                # 4: Calculate the loss
                loss = self.criterion(outs, target)
            # 5: Calculate gradients
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # 6: Performing backpropagation
            with torch.no_grad():
                # 7: Update loss
                running_loss.add(loss.item())
                total_loss.add(loss.item())

                if (i + 1) % self.log_step == 0 or (i + 1) == len(dataloader):
                    self.tsboard.update_loss(
                        "train", running_loss.value()[0], epoch * len(dataloader) + i
                    )
                    running_loss.reset()

                # 8: Update metric
                outs = detach(outs)
                target = detach(target)
                for m in self.metric.values():
                    value = m.calculate(outs, target)
                    m.update(value)

        vprint("+ Training result", self.verbose)
        avg_loss = total_loss.value()[0]
        vprint(("Loss:", avg_loss), self.verbose)
        for m in self.metric.values():
            m.summary()

    @torch.no_grad()
    def val_epoch(self, epoch, dataloader):
        running_loss = meter.AverageValueMeter()
        for m in self.metric.values():
            m.reset()

        self.model.eval()
        vprint("Evaluating........", self.verbose)
        progress_bar = tqdm(dataloader) if self.verbose else dataloader
        for i, x in enumerate(progress_bar):
            if (i > self.iters) and (self.iters != -1):
                break
            # 1: Load inputs and labels
            input_ids = move_to(x["input_ids"], self.device)
            attention_mask = move_to(x["attention_mask"], self.device)
            target = move_to(x["target"], self.device)

            # 2: Get network outputs
            outs = self.model(input_ids, attention_mask)

            # 3: Calculate the loss
            loss = self.criterion(outs, target)

            # 4: Update loss
            running_loss.add(loss.item())
            # 5: Update metric
            outs = detach(outs)
            target = detach(target)
            for m in self.metric.values():
                value = m.calculate(outs, target)
                m.update(value)

        vprint("+ Evaluation result", self.verbose)
        avg_loss = running_loss.value()[0]
        vprint(("Loss:", avg_loss), self.verbose)
        self.val_loss.append(avg_loss)
        self.tsboard.update_loss("val", avg_loss, epoch)

        for k in self.metric.keys():
            m = self.metric[k].value()
            self.metric[k].summary()
            self.val_metric[k].append(m)
            self.tsboard.update_metric("val", k, m, epoch)

    def train(self, train_dataloader, val_dataloader):

        for epoch in range(self.nepochs):
            vprint("\nEpoch {:>3d}".format(epoch), self.verbose)
            vprint("-----------------------------------", self.verbose)

            # 1: Training phase
            self.train_epoch(epoch=epoch, dataloader=train_dataloader)

            # vprint(,self.verbose)

            # 2: Evalutation phase
            if (epoch + 1) % self.val_step == 0:
                # 2: Evaluating model
                with autocast(enabled=self.fp16):
                    self.val_epoch(epoch, dataloader=val_dataloader)
                vprint("-----------------------------------", self.verbose)

                # 3: Learning rate scheduling
                self.scheduler.step(self.val_loss[-1])

                # 4: Saving checkpoints
                if not self.debug:
                    # Get latest val loss here
                    val_loss = self.val_loss[-1]
                    val_metric = {k: m[-1] for k, m in self.val_metric.items()}
                    self.save_checkpoint(epoch, val_loss, val_metric)
