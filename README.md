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
  │
  ├── train_multi.py : A multi-GPU implementation
  │
  ├── train_pl.py : Pytorch-lightning implementation along with Tensorboard logging
  │
  ├── pl_hydra/ - contains all the files pertaining to pytorch-lightning hydra implementation
  │   └──...
  │
  ├──  utils/ - small utility functions
  │    ├── util.py
  │    └── ...
  │
  └── requirements.txt : file to install python dependencies
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

```
# Start training with default parameters: 
python train_simple.py --run_name=test_single

# You can either parameters through commandline, for e.g.:
python train_simple.py -bs=64 -ep=2 --run_name=test_single

# You can also set parameters run_simple.sh file and start the training as following:
source train_simple.py
```

## Multi-GPU implementation
```
# Training with default parameters and 2 GPU: 
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9995 train_multi.py --run_name=test_multi

# You can also pass parameters through commandline (single GPU training), for e.g.:
python -m torch.distributed.launch --nproc_per_node=1 --master_port=9995 train_multi.py -ep=5 --run_name=test_single

# You can also set parameters in run_simple.sh file and start the training as following:
source train_multi.py
```

## Pytorch-lightning implementation
```
# Training with 1 GPU:
python train_pl.py --epochs=5 --run_name=test_pl --gpus=1

# Training with 2 GPUs:
python train_pl.py --epochs=5 --run_name=test_pl --gpus=2
```

```
# Running the Tensorboard:
tensorboard --logdir ./logs/
```

## Pytorch-lightning Hydra implementation
[Tensorboard containing the runs comparing different architectures on CIFAR10](https://tensorboard.dev/experiment/JUrYiGdOQqC0iGNoWtdPlg/#scalars&run=densenet%2F2022-05-06_00-27-19%2Ftensorboard%2Fdensenet&runSelectionState=eyJkZW5zZW5ldC8yMDIyLTA1LTA2XzAwLTI3LTE5L3RlbnNvcmJvYXJkL2RlbnNlbmV0Ijp0cnVlLCJnb29nbGVuZXQvMjAyMi0wNS0wNl8wOC00OS01My90ZW5zb3Jib2FyZC9nb29nbGVuZXQiOnRydWUsInJlc25ldC8yMDIyLTA1LTA2XzEwLTM1LTM5L3RlbnNvcmJvYXJkL3Jlc25ldCI6dHJ1ZSwidmdnLzIwMjItMDUtMDVfMTUtNTYtMDAvdGVuc29yYm9hcmQvdmdnIjp0cnVlLCJ2aXQvMjAyMi0wNS0wNV8xNS0wMS01NS90ZW5zb3Jib2FyZC92aXQiOnRydWV9)
 

## Quickstart
```
# clone project
git clone https://https://github.com/garg-aayush/pytorch-pl-hydra-templates
cd pytorch-pl-hydra-templates

# create conda environment
conda create -n pl_hydra python=3.8
conda activate pl_hydra

# install requirements
pip install -r requirements.txt
```

## Feedback
To give feedback or ask a question or for environment setup issues, you can use the [Github Discussions](https://https://github.com/garg-aayush/pytorch-pl-hydra-templates/discussions).