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


class M2GDataset(Dataset):
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
        label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        V_at = self.data['time_label'][index]
        start_fea = self.data['start_fea'][index]
        cou_fea = self.data['cou_fea'][index]

        # aoi features
        aoi_feature_steps = self.data["aoi_feature_steps"][index]
        aoi_start_steps = self.data["aoi_start_steps"][index]
        aoi_pos_steps = self.data["aoi_pos_steps"][index]
        aoi_len_steps = self.data["aoi_len_steps"][index]
        aoi_idx_steps = self.data["aoi_idx_steps"][index]
        aoi_index_steps = self.data["aoi_index_steps"][index]
        E_static_fea = self.data['E_static_fea'][index]
        E_mask = self.data['E_mask'][index]

        return V, V_reach_mask, label, label_len, V_at, start_fea, cou_fea,\
               aoi_feature_steps, aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_idx_steps, aoi_index_steps, E_static_fea, E_mask

if __name__ == '__main__':
    pass
