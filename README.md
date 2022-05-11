# Template codes for Deep learning with Pytorch
This repo provides different pytorch implementation for training a deep learning model. It uses a simple classification task example for [CIFAR10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) to show:
  1. A simple vanilla, [single-GPU implementation](#single-gpu-implementation)
  2. A [multi-GPU implementation](#multi-gpu-implementation) using Distrbuted data parallel
  3. A [Pytorch-ligtning implementation](#pytorch-lightning-implementation) along with tracking and visualization in TensorBoard
  4. A [Pytorch-ligtning Hydra implementation](#pytorch-lightning-hydra-implementation) for rapid experimentation and prototyping using new models/datasets

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

## Quickstart
<details> 
<summary><b>Folder structure</b></summary>
  
  ```
  pytorch-templates/
  │
  ├── train_simple.py : A single-GPU implementation
  ├── run_simple.py   : bash script to run train_simple.py and pass arguments
  │
  ├── train_multi.py : A multi-GPU implementation
  ├── run_multi.py   : bash script to run train_multi.py and pass arguments
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
  
</details>

<details> 
<summary><b>Setting up the environment</b></summary>

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
</details>
  
## Single-GPU implementation
`train_simple.py` is a very vanilla [pytorch](https://pytorch.org/) implementation that can either run on a CPU or a single GPU. The code uses own simple functions to log different metrics, print out info at run time and save the model at the end of the run. Furthermore, the [Argparse](https://docs.python.org/3/library/argparse.html) module is used to parse the arguments through commandline. 

<details>
<summary><b>Arguments that can be passed through commandline</b></summary>

> Use `python <python_file> -h` to see the available parser arguments for any script. 

```
usage: train_simple.py [-h] --run_name RUN_NAME [--random_seed RANDOM_SEED]
                       [-et EPOCHS_PER_TEST] [-ep EPOCHS] [-bs BATCH_SIZE]
                       [-w NUM_WORKERS] [--learning_rate LEARNING_RATE]
                       [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                       [--gamma GAMMA]

required arguments:
  --run_name RUN_NAME
  
optional arguments:
  -h, --help            show this help message and exit
  --run_name RUN_NAME
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
</details>

<details> 
<summary><b>Running the script</b></summary>
  
```
# Start training with default parameters: 
python train_simple.py --run_name=test_single

# You can either parameters through commandline, for e.g.:
python train_simple.py -bs=64 -ep=2 --run_name=test_single

# You can also set parameters run_simple.sh file and start the training as following:
source train_simple.py
```
  
</details>
 
NOTE: remember to set the data folder path (`DATASET_PATH`) and model checkpoint path (`CHECKPOINT_PATH`) in the `train_simple.py`


## Multi-GPU implementation
`train_multi.py` is a multi-GPU [pytorch](https://pytorch.org/) implementation that  uses Pytorch's [Distributed Data Parallel (DDP)](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for data parallelism. The code is almost similar to You can either run on a CPU or a single GPU or multiple-GPUS. The code is very similar to [single-GPU implementation](#single-gpu-implementation) except the use of DDP and Distributed sampler.

<details>
<summary><b>Arguments that can be passed through commandline</b></summary>

> Use `python <python_file> -h` to see the available parser arguments for any script. 

```
usage: train_multi.py [-h] --run_name RUN_NAME [--random_seed RANDOM_SEED] [-nr LOCAL_RANK]
                      [-et EPOCHS_PER_TEST] [-ep EPOCHS] [-bs BATCH_SIZE] [-w NUM_WORKERS]
                      [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM]
                      [--gamma GAMMA]


required arguments:
  --run_name RUN_NAME
  
optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED
  -nr LOCAL_RANK, --local_rank LOCAL_RANK
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
</details>

<details> 
<summary><b>Running the script</b></summary>
  
```
# Training with default parameters and 2 GPU: 
python -m torch.distributed.launch --nproc_per_node=2 --master_port=9995 train_multi.py --run_name=test_multi

# You can also pass parameters through commandline (single GPU training), for e.g.:
python -m torch.distributed.launch --nproc_per_node=1 --master_port=9995 train_multi.py -ep=5 --run_name=test_multi

# You can also set parameters in run_multi.sh file and start the training as following:
source train_multi.py
```
  
</details>
 
NOTE: remember to set the data folder path (`DATASET_PATH`) and model checkpoint path (`CHECKPOINT_PATH`) in the `train_simple.py`

## Pytorch-lightning implementation
`train_pl.py` is a [pytorch-lightning](https://www.pytorchlightning.ai/) implementation that helps to organize the code neatly and provides lot of logging, metrics and multi-platform run features. The code is organised by creating a separate [Pytorch ligtning module class](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) and a separate [Pyotrch lightning datamodule class](https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html). Moreover, here we log all the metrics, the [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) and validation/test prediction images at each epoch. All this logging info can be viewed using the [Tensorboard](https://www.tensorflow.org/tensorboard).

<details> and a contains all the ta
<summary><b>Commandline arguments</b></summary>

> Use `python <python_file> -h` to see the available parser arguments for any script. 

```
usage: train_pl.py [-h] --run_name RUN_NAME [--random_seed RANDOM_SEED] [-ep EPOCHS] [-bs BATCH_SIZE]
                   [-w NUM_WORKERS] [-g GPUS] [--learning_rate LEARNING_RATE]
                   [--weight_decay WEIGHT_DECAY] [--momentum MOMENTUM] [--gamma GAMMA]

required arguments:
  --run_name RUN_NAME
  
optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED
  -ep EPOCHS, --epochs EPOCHS
                        Total number of training epochs to perform.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
  -w NUM_WORKERS, --num_workers NUM_WORKERS
  -g GPUS, --gpus GPUS
  --learning_rate LEARNING_RATE
                        The initial learning rate for SGD.
  --weight_decay WEIGHT_DECAY
                        Weight deay if we apply some.
  --momentum MOMENTUM   Momentum value in SGD.
  --gamma GAMMA         gamma value for MultiStepLR.
```
</details>

<details> 
<summary><b>Running the script</b></summary>
  
```bash
# Training with 1 GPU:
python train_pl.py --epochs=5 --run_name=test_pl --gpus=1

# Training with 2 GPUs:
python train_pl.py --epochs=5 --run_name=test_pl --gpus=2
```

</details>

<details> 
<summary><b>Starting the Tensorboard</b></summary>

```
tensorboard --logdir ./logs/
```

</details>

NOTE: remember to set the data folder path (`DATASET_PATH`) and model checkpoint path (`CHECKPOINT_PATH`) in the `train_simple.py`

## Pytorch-lightning Hydra implementation
`pl_hydra/` contains all the code pertaining to pl-hydra implementation. This implementation is based on [Ashleve's lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template). The template allows fast experimentation by making the use of [pytorch-lightning](https://www.pytorchlightning.ai) to organize the code and [hydra](https://hydra.cc/) to compose the configuration files that can be used to define different target, pass arguments, etc. for the run. Thus, avoiding the need to maintain multiple configuration files.

<details> 
<summary><b>pl_hydra folder structure</b></summary>

> Modified from [Ashleve's lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)

```
pl_hydra
│
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── local                    <- Local configs
│   ├── log_dir                  <- Logging directory configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── train.yaml             <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── notebooks              <- Jupyter notebooks
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── datamodules              <- Lightning datamodules
│   ├── models                   <- Lightning models
│   ├── utils                    <- Utility scripts
│   ├── vendor                   <- Third party code that cannot be installed using PIP/Conda
│   │
│   └── training_pipeline.py
│
├── train.py              <- Run training
│
├── setup.cfg                 <- Configuration of linters and pytest
└── README.md
```
</details>

The code useds multiple config files to instantiate datamodules, optimizers, etc. and to pass arguments. 

The [train.yaml](pl_hydra/configs/train.yaml) is the main config file that contains default training configuration.
It determines how config is composed when simply executing command `python train.py`.

<details>
<summary><b>Show main project config</b></summary>

```yaml
# @package _global_

# specify here default training configuration
defaults:
  - _self_
  - datamodule: cifar10.yaml
  # for resnet 
  - model : cifar10_resnet.yaml
  - optim: optim_sgd.yaml
  # # for googlenet 
  # - model : cifar10_googlenet.yaml
  # - optim: optim_adam.yaml
  # # for densenet 
  # - model : cifar10_densenet.yaml
  # - optim: optim_adam.yaml
  # for vgg11 
  # - model : cifar10_vgg11.yaml
  # - optim: optim_adam.yaml
  # # for Vit 
  # - model : cifar10_vit.yaml
  # - optim: optim_adam_vit.yaml
  # - callbacks: default.yaml
  - logger: tensorboard.yaml # set logger here or use command line (e.g. `python train.py logger=tensorboard`)
  # - trainer: ddp.yaml
  - trainer: default.yaml
  - log_dir: default.yaml
  # experiment configs allow for version control of specific configurations
  # e.g. best hyperparameters for each combination of model and datamodule
  - experiment: null

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # enable color logging
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog

# default name for the experiment, determines logging folder path
# (you can overwrite this name in experiment configs)
name: "test"

# path to original working directory
# hydra hijacks working directory by changing it to the new log directory
# https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
original_work_dir: ${hydra:runtime.cwd}

# path to folder with data
data_dir: ${original_work_dir}/../../data/

# pretty print config at the start of the run using Rich library
print_config: True

# disable python warnings if they annoy you
ignore_warnings: True

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True

# seed for random number generators in pytorch, numpy and python.random
seed: 100
```
</details>

Apart from the main config, there are separate configs for optimizers, modules, dataloaders and loggers. For example, this is a optimizer config:
<details>
<summary><b>Show example optimizer config</b></summary>

> [pl_hydra/configs/optim/optim_adam.yaml](pl_hydra/configs/optim/optim_adam.yaml)

```yaml
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-3
  weight_decay: 1e-4

use_lr_scheduler: True

lr_scheduler:
  _target_: torch.optim.lr_scheduler.MultiStepLR
  milestones: [90,130]
  gamma: 0.1
```

</details>

This helps to maintain and use different optimizers. In order to use a different optimizer, just specfiy the different optimizer and corresponding parameters in the optim izerconfig file, or else, just write a different optimizer config file and add path to [pl_hydra/configs/train.yaml](pl_hydra/configs/train.yaml).

<details> 
<summary><b>Running the script</b></summary>
  
```
# Note: make sure to go to pl_hydra first
cd pl_hydra

# Training with default parameters:
python train.py

# train on 1 GPU
python train.py trainer.gpus=1

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer.gpus=2 +trainer.strategy=ddp

# train model using googlenet architecture and adam optimizer
python train.py model=googlenet optim=optim_adam
```
  
</details>

Note, make sure to go inside **pl_hydra** folder (`cd pl_hydra`) before running the scripts.

<details> 
<summary><b>Training CIFAR10 using different architectures</b></summary>

> In order to see the ease with which you can experiment, code contains different model architectures ([ResNet](), [GoogeNet](), [VGG](), [DenseNet](), [ViT]()) that can be used to train CIFAR10 and compare the performance. The architectures are defined in [pl_hydra/src/models/components](pl_hydra/src/models/componentspl_hydra/src/models/components).

```
# Note: make sure to go to pl_hydra first
cd pl_hydra

# train model using ResNet
python train.py model=cirfar10_resnet optim=optim_sgd

# train model using GoogleNet
python train.py model=cirfar10_googlenet optim=optim_adam

# train model using DenseNet
python train.py model=cirfar10_densenet optim=optim_adam

# train model using VGG11
python train.py model=cirfar10_vgg11 optim=optim_adam

# train model using ViT
python train.py model=cirfar10_vit optim=optim_adam_vit
```
  
</details>

Note, make sure to go inside **pl_hydra** folder (`cd pl_hydra`) before running the scripts.

[Tensorboard containing the runs comparing different architectures on CIFAR10](https://tensorboard.dev/experiment/JUrYiGdOQqC0iGNoWtdPlg/#scalars&run=densenet%2F2022-05-06_00-27-19%2Ftensorboard%2Fdensenet&runSelectionState=eyJkZW5zZW5ldC8yMDIyLTA1LTA2XzAwLTI3LTE5L3RlbnNvcmJvYXJkL2RlbnNlbmV0Ijp0cnVlLCJnb29nbGVuZXQvMjAyMi0wNS0wNl8wOC00OS01My90ZW5zb3Jib2FyZC9nb29nbGVuZXQiOnRydWUsInJlc25ldC8yMDIyLTA1LTA2XzEwLTM1LTM5L3RlbnNvcmJvYXJkL3Jlc25ldCI6dHJ1ZSwidmdnLzIwMjItMDUtMDVfMTUtNTYtMDAvdGVuc29yYm9hcmQvdmdnIjp0cnVlLCJ2aXQvMjAyMi0wNS0wNV8xNS0wMS01NS90ZW5zb3Jib2FyZC92aXQiOnRydWV9)
 


## Feedback
To give feedback or ask a question or for environment setup issues, you can use the [Github Discussions](https://https://github.com/garg-aayush/pytorch-pl-hydra-templates/discussions).
