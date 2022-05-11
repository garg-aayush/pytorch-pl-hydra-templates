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
```
usage: train_simple.py [-h] --run_name RUN_NAME [--random_seed RANDOM_SEED]
                       [-et EPOCHS_PER_TEST] [-ep EPOCHS] [-bs BATCH_SIZE]
                       [-w NUM_WORKERS] [--learning_rate LEARNING_RATE]
                       [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                       [--gamma GAMMA]

optional arguments:
  -h, --help            show this help message and exit
  --run_name RUN_NAME
  --random_seed RANDOM_SEED
  -et EPOCHS_PER_TEST, --epochs_per_test EPOCHS_PER_TEST
                        Number of epochs per test/val
  -ep EPOCHS, --epochs EPOCHS
                        Total number of training epochs to perform.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  -w NUM_WORKERS, --num_workers NUM_WORKERS
  --learning_rate LEARNING_RATE
                        The initial learning rate for SGD.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --momentum MOMENTUM   Momentum value in SGD.
  --gamma GAMMA         gamma value for MultiStepLR.
```
## Multi-GPU implementation

## Pytorch-lightning implementation

## Pytorch-lightning Hydra implementation

