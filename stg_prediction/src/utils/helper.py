from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import Tensor
import logging
import numpy as np
import pandas as pd
import os
import sys
import pickle
import random
import torch_geometric
from src.utils.scaler import StandardScaler

def get_dataloader(datapath, batch_size,input_dim, output_dim, mode='train'):
    data = {}
    processed = {}
    results = {}
    
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    # we use different the scalers for each node
    scalers = []
    for i in range(data['x_train'].shape[2]):
        scalers.append(StandardScaler(mean=data['x_train'][:,:,i, 0].mean(),
                                      std=data['x_train'][:,:,i, 0].std()))

    # Normalize each node
    for category in ['train', 'val', 'test']:
        for i in range(data['x_train'].shape[2]):
            data['x_' + category][:,:,i, :1] = scalers[i].transform(data['x_' + category][:,:,i, :1])
            data['y_' + category][:,:,i, :1] = scalers[i].transform(data['y_' + category][:,:,i, :1])

        new_x = Tensor(data['x_' + category])[..., :input_dim]
        new_y = Tensor(data['y_' + category])[..., :output_dim]

        processed[category] = TensorDataset(new_x, new_y)

    results['train_loader'] = DataLoader(processed['train'], batch_size, shuffle=True)
    results['val_loader'] = DataLoader(processed['val'], batch_size, shuffle=False)
    results['test_loader'] = DataLoader(processed['test'], batch_size, shuffle=False)

    print('train: {}\t valid: {}\t test:{}'.format(len(results['train_loader'].dataset),
                                                   len(results['val_loader'].dataset),
                                                   len(results['test_loader'].dataset)))
    results['scalers'] = scalers
    return results


def check_device(device=None):
    if device is None:
        print("`device` is missing, try to train and evaluate the model on default device.")
        if torch.cuda.is_available():
            print("cuda device is available, place the model on the device.")
            return torch.device("cuda")
        else:
            print("cuda device is not available, place the model on cpu.")
            return torch.device("cpu")
    else:
        if isinstance(device, torch.device):
            return device
        else:
            return torch.device(device)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def get_num_nodes(dataset):
    print(dataset)
    d = {'Delivery_SH':30,
        'Delivery_HZ':31,
        'Delivery_CQ':30,
        'Delivery_YT':30,
        'Delivery_JL':14,}
    assert dataset in d.keys()
    return d[dataset]

def get_null_value(dataset):
    d = {'Delivery':-1.0}
    assert dataset[:8] in d.keys()
    return d[dataset[:8]]




