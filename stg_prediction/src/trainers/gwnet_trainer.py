import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from src.utils.logging import get_logger
from src.base.trainer import BaseTrainer
from src.utils import graph_algo


class GWNET_Trainer(BaseTrainer):
    def __init__(self, **args):
        super(GWNET_Trainer, self).__init__(**args)
        self._optimizer = Adam(self.model.parameters(), self._base_lr)
        self._supports = self._calculate_supports(args['adj_mat'], args['filter_type'])
        
    def _calculate_supports(self, adj_mat, filter_type):
        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)
        
        if filter_type == "scalap":
            supports =[graph_algo.calculate_scaled_laplacian(new_adj).todense()]
        elif filter_type == "normlap":
            supports =[graph_algo.calculate_normalized_laplacian(
                new_adj).astype(np.float32).todense()]
        elif filter_type == "symnadj":
            supports =[graph_algo.sym_adj(new_adj)]
        elif filter_type == "transition":
            supports =[graph_algo.asym_adj(new_adj)]
        elif filter_type == "doubletransition":
            supports =[graph_algo.asym_adj(new_adj), 
                       graph_algo.asym_adj(np.transpose(new_adj))]
        elif filter_type == "identity":
            supports =[np.diag(np.ones(new_adj.shape[0])).astype(np.float32)]
        else:
            error = 0
            assert error, "adj type not defined"
        supports = [torch.tensor(i).cuda() for i in supports]
        return supports