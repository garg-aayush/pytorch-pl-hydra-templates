from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional import confusion_matrix

from src.models.components.resnet import ResNet
#import torch.optim as optim
import torch.nn as nn

import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from src.utils.plotter import plot_cm, plot_preds

import hydra

## classes
name_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
num_classes = len(name_classes)
data_mean = [0.49421428, 0.48513139, 0.45040909]
data_std = [0.24665252, 0.24289226, 0.26159238]

class CIFAR10LitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self,*args,**kwargs):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        print('here')
        
        self.save_hyperparameters()
        # network
        self.net = hydra.utils.instantiate(self.hparams['net'])

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
                                                                                                           
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        # plot the confusion matrix at the end of each epoch
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        
        # plot confusion matrix
        cm = confusion_matrix(targets, preds, num_classes)
        fig_ = plot_cm(cm, name_classes)
        plt.close(fig_)
        self.logger.experiment.add_figure("confusion_matrix_train", fig_, self.current_epoch)
        
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # plot figures
        if batch_idx == 0:
            images, _ = batch
            fig_ = plot_preds(images.cpu().numpy(), 
                            targets.cpu().numpy(), 
                            preds.cpu().numpy(), 
                            name_classes,
                            nimg=32,
                            ncols=8,
                            data_mean=data_mean,
                            data_std=data_std)
            self.logger.experiment.add_figure(
                                    "examples_val_batch_idx_" + str(batch_idx),
                                    fig_, 
                                    self.current_epoch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        #print(preds)
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

        # plot the confusion matrix at the end of each epoch
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        cm = confusion_matrix(targets, preds, num_classes)
        fig_ = plot_cm(cm, name_classes)
        plt.close(fig_)
        self.logger.experiment.add_figure("confusion_matrix_val", fig_, self.current_epoch)

        
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        
        # plot figures
        if batch_idx == 0:
            images, _ = batch
            fig_ = plot_preds(images.cpu().numpy(), 
                            targets.cpu().numpy(), 
                            preds.cpu().numpy(), 
                            name_classes,
                            nimg=32,
                            ncols=8,
                            data_mean=data_mean,
                            data_std=data_std)
            self.logger.experiment.add_figure(
                                    "examples_test_batch_idx_" + str(batch_idx),
                                    fig_, 
                                    self.current_epoch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        # plot the confusion matrix at the end of each epoch
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['targets'] for tmp in outputs])
        cm = confusion_matrix(targets, preds, num_classes)
        fig_ = plot_cm(cm, name_classes)
        plt.close(fig_)
        
        self.logger.experiment.add_figure("confusion_matrix_test", fig_, self.current_epoch)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

   
    def configure_optimizers(self):
        optimizer=hydra.utils.instantiate(
                                        self.hparams.optim["optimizer"],
                                        params=self.net.parameters()
                                        )
        
        if(self.hparams.optim['use_lr_scheduler']==True):
            scheduler=hydra.utils.instantiate(
                                            self.hparams.optim['lr_scheduler'],
                                            optimizer=optimizer
                                            )
            return [optimizer],[scheduler]
        else:
            return optimizer