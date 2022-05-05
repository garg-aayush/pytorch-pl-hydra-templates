
## See https://gist.github.com/sgraaf/5b0caa3a320f28c27c12b5efeb35aa4c
## Standard libraries
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
import torch.distributed as dist

# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms

# load network
from utils.ResNet import ResNet

# load helper functions 
from utils.helper import set_random_seed, print_info, evaluate

# Set the visible GPUs, in case of multi-GPU device, otherwise comment it
# you can use `nvidia-smi` in terminal to see the available GPUS
os.environ["CUDA_VISIBLE_DEVICES"]="13,14"


######################################################################
# Set the Global values
######################################################################
# Transform argument
IMAGE_SIZE = (32,32)  # H X W
SCALE_BOUNDS = (0.8,1.0) # lower & upper bounds
ASPECT_BOUNDS = (0.9,1.1) # lower & upper bounds
                                     
# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = "../data"

# Path to the folder where the models will be saved
CHECKPOINT_PATH = "../saved_models/multi/"
os.makedirs(CHECKPOINT_PATH,exist_ok=True)

# SAVE STATS
SAVE_STATS = True
stats_file = 'simple_stats.json'
SAVE_PARAMS = True
params_file = 'simple_params.json'
SAVE_MODEL = True
model_filename = 'simple_checkpoint.h5'



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
    parser.add_argument('-nr','--local_rank',default=0,type=int)
                   
    parser.add_argument('-et','--epochs_per_test',default=1,type=int,
                        help="Number of epochs per test/val")
    parser.add_argument('-ep','--epochs',default=150,type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('-bs','--batch_size',default=128,type=int)
    parser.add_argument('-w','--num_workers',default=4,type=int)
    
    
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
    local_rank = args.local_rank

    epochs=args.epochs
    epochs_per_test=args.epochs_per_test
    batch_size=args.batch_size
    num_workers=args.num_workers
    
    lr = args.learning_rate
    weight_decay = args.weight_decay
    momentum = args.momentum
    gamma = args.gamma
    milestones = [90, 130]

    # print input arguments
    print(f'Run name : {run_name}')
    print(f'Random seed: {random_seed}')
    print(f'Local rank: {local_rank}')
    print(f'Num of training epochs: {epochs}')
    print(f'Num of epochs per test: {epochs_per_test}')
    print(f'Batch size: {batch_size}')
    print(f'Num workers: {num_workers}')
    
    print(f'Learning rate: {lr}')
    print(f'weight_decay: {weight_decay}')
    print(f'momentum: {momentum}')
    print(f'gamma: {gamma}')
    

    ######################################################################
    # GPU configuration
    ######################################################################
    # initialize the distributed process group.
    dist.init_process_group(backend='nccl')
    # set the random seed
    set_random_seed(random_seed)
 
    # set device for this run.
    device=torch.device("cuda:{}".format(local_rank))
    print("\nUsing device: ", device)
    
    ######################################################################
    # Train and val datasets
    ######################################################################
    print("\nUsing device: ", device)

    # Download the dataset if required
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
    DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
    DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
    print("\nData mean", DATA_MEANS)
    print("Data std", DATA_STD)


    #Define transforms
    # For training, we add some augmentation. Network is too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop(IMAGE_SIZE,scale=SCALE_BOUNDS,ratio=ASPECT_BOUNDS),
                                      transforms.ToTensor(),
                                      transforms.Normalize(DATA_MEANS, DATA_STD)
                                     ])
    val_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(DATA_MEANS, DATA_STD)
                                     ])
    

    # Loading the training dataset.
    print('\nApply trainsforms')
    print('Apply following transforms to the train dataset')
    print(train_transform)
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    
    # Loading the val dataset
    print('Apply following transforms to the val dataset')
    print(val_transform)
    val_dataset = CIFAR10(root=DATASET_PATH, train=False, transform=val_transform, download=True)
    
    
    # We define a set of data loaders that we can use for various purposes later.
    train_sampler=torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True, num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    
    print(f'\nNumber of train examples: {len(train_loader.dataset)}')
    print(f'Number of val examples: {len(val_loader.dataset)}')
    
    # Check the normalization
    print('\nCheck the normalization')
    imgs, _ = next(iter(train_loader))
    print("Img_size", imgs.shape)
    # returns mean and std for each channel (channel dim is 1)
    print("Batch mean", imgs.mean(dim=[0,2,3]))
    print("Batch std", imgs.std(dim=[0,2,3]))

    
    ######################################################################
    # Setup the model
    ######################################################################
    # make model, send to device
    net=ResNet().to(device)
    net=nn.parallel.DistributedDataParallel(net,device_ids=[device],output_device=device)

    # loss function 
    lossfunc=nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer=optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler=optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


    ######################################################################
    # Training
    ######################################################################
    # var to save the history
    start = 0 
    train_loss_epochs = []
    train_loss_history = []
    train_acc_history = []
    
    val_loss_history = []
    val_acc_history = []
    
    learning_rate_history = []
    
    best_loss = float('inf')

    # Number of steps per epoch
    total_train_steps = len(train_loader)
    total_val_steps = len(val_loader)

    print('\nSteps per epoch')
    print(f'Training steps per epoch: {total_train_steps}')
    print(f'Validation steps per epoch: {total_val_steps}')
    print()

    # iter over epochs
    for epoch in range(start,epochs):
        if local_rank ==0:
            print(f'Epoch {epoch+1}/{epochs}')
        
        #set it to train
        net.train()
        
        dist.barrier()
        epoch_start_time=time.time()
        correct = 0
        train_loss=0
        train_acc=0

        for _, (data, label) in enumerate(train_loader):
            data,label=data.to(device), label.to(device)
            
            # 
            net.zero_grad()
            optimizer.zero_grad()
            
            #Forward pass
            output=net(data)
            
            # calculate loss
            loss=lossfunc(output,label)
            
            # backward and optimize
            loss.backward()
            optimizer.step()
            
            # accuracy
            correct = (output.argmax(dim=-1) == label).float().mean()
            
            # each step loss
            # step_loss.append(loss.item)
            train_loss += loss.item()/total_train_steps
            train_acc += 100.0 * (correct.item()/total_train_steps)
            
        ######################################################################################################
        # Print and save train loss info
        ######################################################################################################
        if local_rank ==0:
            train_loss_history.append(train_loss)
            train_acc_history.append(train_acc)
            train_loss_epochs.append(epoch)
            
            elapsed_time = time.time() - epoch_start_time 
            print_info('train', loss=train_loss_history[-1], acc=train_acc_history[-1], elapsed_time=elapsed_time)


        ######################################################################
        # Val evaluation at each epochs_per_test
        ######################################################################
        if ((epoch>=0 and epoch%epochs_per_test==0) or (epoch==args.epochs-1)):
            # val_data
            start = time.time()
            val_loss, val_acc = evaluate(model=net,device=device,loader=val_loader,lossfunction=lossfunc)
            elapsed_time = time.time() - start
            
            if local_rank == 0:
                val_loss_history.append(val_loss)
                val_acc_history.append(val_acc)
                print_info('val', loss=val_loss_history[-1], acc=val_acc_history[-1], elapsed_time=elapsed_time)

               
        scheduler.step()
        if local_rank == 0:
            learning_rate_history.append(optimizer.param_groups[0]['lr'])
        print(f"learning rate={optimizer.param_groups[0]['lr']} at gpu {device}")
        

        ######################################################################
        # Save the model and optimizer state (best)
        ###################################################################### 
        # overwrite the model and optimizer state whenever test_loss decreases 
        if local_rank == 0:
            if ((val_loss < best_loss) or (epoch==args.epochs-1)):
                print(f'Saving model. Test loss improved from {best_loss:.2f} to {val_loss:.2f}')
                best_loss=val_loss
                    
                #save model
                output_file=os.path.join(CHECKPOINT_PATH, model_filename+'-model')
                torch.save(net.state_dict(),output_file)

                # save optimizer state
                output_file=os.path.join(CHECKPOINT_PATH, model_filename+'-optim')
                torch.save(optimizer.state_dict(),output_file)


    ######################################################################
    # Save the loss and accuracy history
    ###################################################################### 
    if local_rank == 0:
        if SAVE_STATS:
            stats = {'epochs': train_loss_epochs,
                'train_loss_history': train_loss_history,
                'train_acc_history': train_acc_history,
                'val_loss_history': val_loss_history,
                'val_acc_history': val_acc_history
                }
        
            # Writing as json file
            with open(CHECKPOINT_PATH + stats_file, "w") as outfile:
                json.dump(stats, outfile, indent=4)


    ######################################################################
    # Save the parameters for this run
    ###################################################################### 
        if SAVE_PARAMS:
            stats = {'run_name': run_name,
                'random_seed': random_seed,
                'epochs': epochs,
                'epochs_per_test': epochs_per_test,
                'batch_size': batch_size,
                'num_workers': num_workers,
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'gamma': gamma,
                'milestones': milestones
                }
        
            # Writing as json file
            with open(CHECKPOINT_PATH + params_file, "w") as outfile:
                json.dump(stats, outfile, indent=4)
        

if __name__=='__main__':
    main()