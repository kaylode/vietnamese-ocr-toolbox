# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
import os
import cv2
import shutil
import numpy as np
import traceback
import time
from tqdm import tqdm
import torch
import torchvision.utils as vutils
from torchvision import transforms
from ..metrics import runningScore, cal_text_score, cal_kernel_score, cal_recall_precison_f1
from ..base import BaseTrainer
from ..predict import PAN

class Trainer(BaseTrainer):
    def __init__(self, args, config, model, criterion, train_loader, val_loader, metric, weights_init=None):
        super(Trainer, self).__init__(args, config, model, criterion, metric, weights_init)
        
        self.show_images_interval = args.val_interval
        self.save_interval = args.save_interval
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_len = len(train_loader)
        self.val_loader_len = len(val_loader)
        
        self.logger.info('train dataset has {} samples,{} in dataloader'.format(self.train_loader.dataset_len,
                                                                                self.train_loader_len))

        self.logger.info('val dataset has {} samples,{} in dataloader'.format(self.val_loader.dataset_len,
                                                                                self.val_loader_len))

        self.logger.info(f"number of trainable parameters: {model.trainable_parameters()}")

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']
        for i, (images, labels, training_masks) in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            # 数据进行转换和丢到gpu
            cur_batch_size = images.size()[0]
            images, labels, training_masks = images.to(self.device), labels.to(self.device), training_masks.to(
                self.device)

            preds = self.model(images)
            loss_all, loss_tex, loss_ker, loss_agg, loss_dis = self.criterion(preds, labels, training_masks)
            # backward
            self.optimizer.zero_grad()
            loss_all.backward()
            self.optimizer.step()

            # acc iou
            score_text = cal_text_score(preds[:, 0, :, :], labels[:, 0, :, :], training_masks, running_metric_text)
            score_kernel = cal_kernel_score(preds[:, 1, :, :], labels[:, 1, :, :], labels[:, 0, :, :], training_masks,
                                            running_metric_kernel)

            # loss 和 acc 记录到日志
            loss_all = loss_all.item()
            loss_tex = loss_tex.item()
            loss_ker = loss_ker.item()
            loss_agg = loss_agg.item()
            loss_dis = loss_dis.item()
            train_loss += loss_all
            acc = score_text['Mean Acc']
            iou_text = score_text['Mean IoU']
            iou_kernel = score_kernel['Mean IoU']

            if (i + 1) % self.display_interval == 0:
                batch_time = time.time() - batch_start
                self.logger.info(
                    '[{}/{}], [{}/{}], global_step: {}, Speed: {:.1f} samples/sec, acc: {:.4f}, iou_text: {:.4f}, iou_kernel: {:.4f}, loss_all: {:.4f}, loss_tex: {:.4f}, loss_ker: {:.4f}, loss_agg: {:.4f}, loss_dis: {:.4f}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                                            self.display_interval * cur_batch_size / batch_time, acc, iou_text,
                        iou_kernel, loss_all, loss_tex, loss_ker, loss_agg, loss_dis, lr, batch_time))
                batch_start = time.time()

            if (i + 1) % self.save_interval == 0:
                net_save_path = f"{self.checkpoint_dir}/PANNet_last.pth"
                self._save_checkpoint(epoch, net_save_path, save_best=False)
            
            # write tensorboard
            self.writer.add_scalar('TRAIN/LOSS/loss_all', loss_all, self.global_step)
            self.writer.add_scalar('TRAIN/LOSS/loss_tex', loss_tex, self.global_step)
            self.writer.add_scalar('TRAIN/LOSS/loss_ker', loss_ker, self.global_step)
            self.writer.add_scalar('TRAIN/LOSS/loss_agg', loss_agg, self.global_step)
            self.writer.add_scalar('TRAIN/LOSS/loss_dis', loss_dis, self.global_step)
            self.writer.add_scalar('TRAIN/ACC_IOU/acc', acc, self.global_step)
            self.writer.add_scalar('TRAIN/ACC_IOU/iou_text', iou_text, self.global_step)
            self.writer.add_scalar('TRAIN/ACC_IOU/iou_kernel', iou_kernel, self.global_step)
            self.writer.add_scalar('TRAIN/lr', lr, self.global_step)
            
            if (i + 1) % self.save_interval == 0:
                # show images on tensorboard
                self.writer.add_images('TRAIN/imgs', images, self.global_step)
                # text kernel and training_masks
                gt_texts, gt_kernels = labels[:, 0, :, :], labels[:, 1, :, :]
                gt_texts[gt_texts <= 0.5] = 0
                gt_texts[gt_texts > 0.5] = 1
                gt_kernels[gt_kernels <= 0.5] = 0
                gt_kernels[gt_kernels > 0.5] = 1
                show_label = torch.cat([gt_texts, gt_kernels, training_masks.float()])
                show_label = vutils.make_grid(show_label.unsqueeze(1), nrow=cur_batch_size, normalize=False,
                                                padding=20,
                                                pad_value=1)
                self.writer.add_image('TRAIN/gt', show_label, self.global_step)
                # model output
                preds[:, :2, :, :] = torch.sigmoid(preds[:, :2, :, :])
                show_pred = torch.cat([preds[:, 0, :, :], preds[:, 1, :, :]])
                show_pred = vutils.make_grid(show_pred.unsqueeze(1), nrow=cur_batch_size, normalize=False,
                                                padding=20,
                                                pad_value=1)
                self.writer.add_image('TRAIN/preds', show_pred, self.global_step)

        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _val_epoch(self, epoch):
        self.model.eval()
        val_loss = 0.
        acc = 0.
        iou_text = 0.
        iou_kernel = 0.
        running_metric_text = runningScore(2)
        running_metric_kernel = runningScore(2)
        
        epoch_start = time.time()

        model = PAN(self.config, state_dict=self.model.state_dict())
        self.metric.update(model)    
        metric_dict = self.metric.value()
        model = None

        with torch.no_grad():
            for i, (images, labels, training_masks) in enumerate(tqdm(self.val_loader)):
                
                images, labels, training_masks = images.to(self.device), labels.to(self.device), training_masks.to(
                    self.device)

                preds = self.model(images)
                loss_all, loss_tex, loss_ker, loss_agg, loss_dis = self.criterion(preds, labels, training_masks)
            
                # acc iou
                score_text = cal_text_score(preds[:, 0, :, :], labels[:, 0, :, :], training_masks, running_metric_text)
                score_kernel = cal_kernel_score(preds[:, 1, :, :], labels[:, 1, :, :], labels[:, 0, :, :], training_masks,
                                                running_metric_kernel)

                val_loss += loss_all.item()
                acc += score_text['Mean Acc']
                iou_text += score_text['Mean IoU']
                iou_kernel += score_kernel['Mean IoU']

        epoch_end = time.time()

        acc = acc*1.0 / self.val_loader_len
        iou_text = iou_text*1.0 / self.val_loader_len
        val_loss = val_loss*1.0 / self.val_loader_len
        iou_kernel = iou_kernel*1.0 / self.val_loader_len

        self.logger.info(
            '[{}/{}],  val_acc: {:.4f}, val_iou_text: {:.4f}, val_iou_kernel: {:.4f}, val_loss_all: {:.4f}, time:{:.2f}'.format(
                epoch, self.epochs, acc, iou_text, iou_kernel, loss_all, epoch_end-epoch_start))
        
        if acc>self.best_acc:
            self.best_acc=acc
            net_save_path = f"{self.checkpoint_dir}/PANNet_best_acc.pth"
            self._save_checkpoint(epoch, net_save_path, save_best=False)
        if metric_dict['MAP'] > self.best_map:
            self.best_map=metric_dict['MAP']
            net_save_path = f"{self.checkpoint_dir}/PANNet_best_map.pth"
            self._save_checkpoint(epoch, net_save_path, save_best=False)
            
        return acc, iou_text, iou_kernel


    def _on_epoch_finish(self):
        self.logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))
        if self.epoch_result['epoch'] % self.save_interval == 0:
            net_save_path = f"{self.checkpoint_dir}/PANNet_{self.epoch_result['epoch']}.pth"
            save_best = False
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger.info('{}:{}'.format(k, v))
        self.logger.info('finish train')
