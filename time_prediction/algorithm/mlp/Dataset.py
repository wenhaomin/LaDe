# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import json
from torch import Tensor, dtype
from torch.utils import data
from typing import Tuple, List, Dict
from torch.utils.data import Dataset
from random import shuffle, random, seed


class MLPDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict, #parameters dict
    )->None:
        super().__init__()
        if mode not in ["train", "val", "test"]:  # "validate"
            raise ValueError
        path_key = {'train':'train_path', 'val':'val_path','test':'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['label_len'])

    def __getitem__(self, index):

        V = self.data['V'][index]
        V_reach_mask = self.data['V_reach_mask'][index]
        route_label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        time_label = self.data['time_label'][index]
        start_fea = self.data['start_fea'][index]
        cou_fea = self.data['cou_fea'][index]

        return V, V_reach_mask, route_label, label_len, time_label, start_fea, cou_fea

if __name__ == '__main__':
    pass
