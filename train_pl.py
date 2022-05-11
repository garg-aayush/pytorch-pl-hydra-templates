
## Standard libraries
from typing import Any, List, Optional, Tuple
import os
import time
import json
import numpy as np 
import random
from types import SimpleNamespace
import argparse

## PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Pytorch lightning
import pytorch_lightning as pl
# Callbacks 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping, RichModelSummary,RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
# Metrics
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.functional import confusion_matrix

# plotting
import matplotlib.pylab as plt

# load network
from utils.ResNet import ResNet

# load helper functions 
from utils.helper import set_random_seed, print_info, evaluate
from utils.plotter import plot_cm, plot_preds

# # Set the visible GPUs, in case of multi-GPU device, otherwise comment it
# # you can use `nvidia-smi` in terminal to see the available GPUS
# os.environ["CUDA_VISIBLE_DEVICES"]="13,14,15,16"

######################################################################
# Set the Global values
######################################################################
# Transform argument
IMAGE_SIZE = (32,32)  # H X W
SCALE_BOUNDS = (0.8,1.0) # lower & upper bounds
ASPECT_BOUNDS = (0.9,1.1) # lower & upper bounds
                                     
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"
LOG_PATH = "./logs"
os.makedirs(CHECKPOINT_PATH,exist_ok=True)

# Path to the folder where the models will be saved
CHECKPOINT_PATH = "../saved_models/pl/"
os.makedirs(CHECKPOINT_PATH,exist_ok=True)

# Set the device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## classes
name_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
num_classes = len(name_classes)
data_mean = [0.49421428, 0.48513139, 0.45040909]
data_std = [0.24665252, 0.24289226, 0.26159238]


######################################################################
# Pytorch lightning Dataclass
######################################################################
class CIFAR10DataModule(pl.LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        data_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        data_std: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        image_size: Tuple[int, int] = (32, 32), 
        scale_bounds: Tuple[float, float] = (0.8, 1.0),
        aspect_bounds: Tuple[float, float] = (0.9, 1.1)
        ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(image_size,scale=scale_bounds,ratio=aspect_bounds),
                                        transforms.ToTensor(),
                                        transforms.Normalize(data_mean, data_std)
                                     ])

        self.test_transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(data_mean, data_std)
                                     ])
    
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

   
    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir,
                            train=True, 
                            transform=self.train_transforms)
            valset = CIFAR10(self.hparams.data_dir,
                            train=False, 
                            transform=self.test_transforms)
            testset = CIFAR10(self.hparams.data_dir,
                            train=False, 
                            transform=self.test_transforms)
                            
            self.data_train = trainset
            self.data_val = valset
            self.data_test = testset

        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            drop_last=False
        )


######################################################################
# Pytorch lightning model class
######################################################################
class CIFARModule(pl.LightningModule):
    def __init__(self, optimizer_hparams, scheduler_hparams):
        """
        Inputs:
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
            scheduler_hparams - Hyperparameters for the scheduler, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = ResNet()
        # Create loss module
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

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # We will use SGD optimizer
        optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, **self.hparams.scheduler_hparams)
        return [optimizer], [scheduler]

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        loss, preds, targets = self.step(batch)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
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

    def validation_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
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
        
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        acc = self.val_acc(preds, targets)
        # By default logs it per epoch (weighted average over batches)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

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


    def test_step(self, batch, batch_idx):
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
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
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



######################################################################
# PL Trainer 
######################################################################
def train_model(dm=None, save_name=None, 
                gpus=1, strategy=None, sync_batchnorm=False,
                max_epochs=150, device='cpu', logger=None,
                **kwargs):
    
    if save_name is None:
        save_name = 'model'
    
    callbacks = [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val/acc"), 
                LearningRateMonitor("epoch"),
                RichModelSummary(max_depth=-1),
                RichProgressBar()
                ]

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
                        gpus=gpus if str(device)=="cuda" else 0,
                        strategy=strategy,
                        sync_batchnorm=sync_batchnorm,
                        max_epochs=max_epochs,
                        callbacks=callbacks,
                        logger=logger,
                        progress_bar_refresh_rate=1)
    
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    pl.seed_everything(42) # To be reproducable
    model = CIFARModule(**kwargs)
    trainer.fit(model, datamodule=dm)
        
    # # Test best model on validation
    trainer.test(model, datamodule=dm)
    
    return


######################################################################
# Main function
######################################################################
def main():
    ######################################################################
    # input cmdline arguments
    ######################################################################
    parser=argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--run_name',required=True,type=str)
    parser.add_argument('--random_seed',default=42,type=int)             
    parser.add_argument('-ep','--epochs',default=150,type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('-bs','--batch_size',default=128,type=int)
    parser.add_argument('-w','--num_workers',default=4,type=int)
    parser.add_argument('-g','--gpus',default=1,type=int)
    
    # optimizer parameters
    parser.add_argument("--learning_rate", default=0.1, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Momentum value in SGD.")
    parser.add_argument("--gamma", default=0.1, type=float,
                        help="gamma value for MultiStepLR.")
    
    args=parser.parse_args()
    
    run_name=args.run_name
    random_seed=args.random_seed
    
    epochs=args.epochs
    batch_size=args.batch_size
    num_workers=args.num_workers
    gpus=args.gpus
    
    lr = args.learning_rate
    weight_decay = args.weight_decay
    momentum = args.momentum
    gamma = args.gamma
    milestones = [90, 130]

    # print input arguments
    print(f'Run name : {run_name}')
    print(f'Random seed: {random_seed}')
    
    print(f'Num of training epochs: {epochs}')
    print(f'Batch size: {batch_size}')
    print(f'Num workers: {num_workers}')
    print(f'Num of GPUs: {gpus}')
    
    print(f'Learning rate: {lr}')
    print(f'weight_decay: {weight_decay}')
    print(f'momentum: {momentum}')
    print(f'gamma: {gamma}')
    

    # Start the logger
    logger = TensorBoardLogger(save_dir=LOG_PATH, name=run_name)
    # set the random seed
    pl.seed_everything(random_seed)

    ######################################################################
    # Train and val datasets
    ######################################################################
    print("\nUsing device: ", device)

    dm = CIFAR10DataModule(data_dir=DATASET_PATH,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            data_mean=data_mean,
                            data_std=data_std,
                            image_size=IMAGE_SIZE,
                            scale_bounds=SCALE_BOUNDS,
                            aspect_bounds=ASPECT_BOUNDS)

    
    ######################################################################
    # Training
    ######################################################################
    train_model(dm=dm,
                logger=logger, save_name='Resnet',
                gpus=gpus, strategy='ddp', sync_batchnorm=True,
                max_epochs=epochs, device=device,
                optimizer_hparams={"lr": lr, 
                                    "momentum": momentum,
                                    "weight_decay": weight_decay},
                scheduler_hparams={"milestones": milestones,
                                    "gamma": gamma})
    
if __name__=='__main__':
    main()