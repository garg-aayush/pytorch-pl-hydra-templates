# Template codes for Deep learning with Pytorch
This repo provides different pytorch implementation for training a deep learning model. It uses a simple classification task example for [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to show:
  1. A simple vanilla, [single-GPU implementation](#single-gpu-implementation)
  2. A [multi-GPU implementation](#multi-gpu-implementation) using Distrbuted data parallel
  3. A [Pytorch-ligtning implementation](#pytorch-lightning-implementation) along with tracking and visualization in TensorBoard
  4. A [Pytorch-ligtning Hydra implementation](#pytorch-lightning-hydra-implementation) for rapid experimentation and prototyping using new models/datasets

## Folder Structure
  ```
  pytorch-templates/
  │
  ├── train_simple.py : A single-GPU implementation
  |
  ├── train_multi.py : A multi-GPU implementation
  │
  ├── train_pl.py : Pytorch-lightning implementation along with Tensorboard logging
  │
  ├── train_pl.ipynb : Jupyter notebook for Pytorch-lightning implementation along with Tensorboard logging
  │
  ├── pl_hydra/ - contains all the files pertaining to pytorch-lightning hydra implementation
  │   └──data_loaders.py
  |
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```
  
## Single-GPU implementation

## Multi-GPU implementation

## Pytorch-lightning implementation

## Pytorch-lightning Hydra implementation

