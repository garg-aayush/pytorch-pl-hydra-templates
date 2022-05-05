import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
import random


######################################################################
# Helper functions
######################################################################

# Set the random seed
def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# print info
def print_info(step_type='train', loss=0, acc=0, elapsed_time=0):
    epoch_type = step_type
    print(f'{epoch_type}_loss: {loss:.5f},  {epoch_type}_acc: {acc:.2f}%  epoch_time: {elapsed_time:.2f}s')


# evaluate on the trained model
def evaluate(model=None,device=None,loader=None,lossfunction=None):
    model.eval()
    
    loss=0
    acc=0

    total_steps = len(loader)

    for i, (data, label) in enumerate(loader):
        data,label=data.to(device), label.to(device)
        #Forward pass
        output=model(data)
        # calculate loss
        err=lossfunction(output,label)
        
        # accuracy
        correct = (output.argmax(dim=-1) == label).float().mean()
            
        # each step loss
        loss += err.item()/total_steps
        acc += 100.0 * (correct.item()/total_steps)
    
    return loss, acc