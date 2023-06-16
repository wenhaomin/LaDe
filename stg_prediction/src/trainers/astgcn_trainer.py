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


class ASTGCN_Trainer(BaseTrainer):
    def __init__(self, **args):
        super(ASTGCN_Trainer, self).__init__(**args)
        self._optimizer = Adam(self.model.parameters(), self._base_lr)
        self._supports = []