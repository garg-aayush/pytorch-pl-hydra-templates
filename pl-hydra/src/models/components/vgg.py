'''
Modified from https://github.com/chengyangfu/pytorch-vgg-cifar10.git
'''

## Standard libraries
import os

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from types import SimpleNamespace


act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}


import math

import torch.nn as nn
import torch.nn.init as init


class CnnBlock(nn.Module):

    def __init__(self, c_in, c_out, act_fn):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_out - Number of output feature maps
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            act_fn()
        )

    def forward(self, x):
        return self.conv(x)

class VGG11(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self,
                num_classes: int=10,
                act_fn_name = "relu",
                **kwargs):
        super().__init__()
        self.hparams = SimpleNamespace(num_classes=num_classes,
                                        act_fn_name=act_fn_name)
        #print(self.hparams)
        self._create_network()
        self._init_params()


    def _create_network(self):
        
        # Creating the features map
        self.vgg_blocks = nn.Sequential(
            CnnBlock(3, 64, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CnnBlock(64, 128, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CnnBlock(128, 256, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            CnnBlock(256, 256, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CnnBlock(256, 512, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            CnnBlock(512, 512, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CnnBlock(512, 512, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            CnnBlock(512, 512, act_fn=act_fn_by_name[self.hparams.act_fn_name]),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Mapping to classification output
        self.output_net = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(0.4),
                                        nn.Linear(512, 512),
                                        act_fn_by_name[self.hparams.act_fn_name](),
                                        nn.Dropout(0.4),
                                        nn.Linear(512, 512),
                                        act_fn_by_name[self.hparams.act_fn_name](),
                                        nn.Linear(512, self.hparams.num_classes),
                                        )

    def _init_params(self):
        # Based on our discussion in Tutorial 4, we should initialize the convolutions according to the activation function
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    nonlinearity=self.hparams.act_fn_name)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.vgg_blocks(x)
        x = self.output_net(x)
        return x