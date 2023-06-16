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
from torch.optim import RMSprop

from src.utils.logging import get_logger
from src.base.trainer import BaseTrainer
from src.utils import graph_algo
from src.utils.metrics import masked_rmse


class STGCN_Trainer(BaseTrainer):
    def __init__(self, **args):
        super(STGCN_Trainer, self).__init__(**args)
        # self._optimizer = RMSprop(self.model.parameters(), base_lr)
        # self._lr_scheduler = MultiStepLR(self.optimizer,
        #                                  steps,
        #                                  gamma=lr_decay_ratio)
        # self._loss_fn = masked_rmse
        self._supports = self._calculate_supports(args['adj_mat'], args['filter_type'])
        
    def _calculate_supports(self, adj_mat, filter_type):

        num_nodes = adj_mat.shape[0]
        new_adj = adj_mat + np.eye(num_nodes)

        if filter_type == "identity":
            supports = np.diag(np.ones(new_adj.shape[0])).astype(np.float32)
            supports = Tensor(supports).cuda()
        else:
            scaled_adj = graph_algo.calculate_scaled_laplacian(new_adj).todense()
            cheb_poly_adj = graph_algo.calculate_cheb_poly(scaled_adj, 3)
            supports = Tensor(cheb_poly_adj).cuda()
        return supports