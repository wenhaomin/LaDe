# -*- coding: utf-8 -*-
import numpy as np
from torch.utils.data import Dataset

class FDNetDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,
    )->None:
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train':'train_path', 'val':'val_path','test':'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['label_len'])

    def __getitem__(self, index):
        E_static_fea = self.data['E_static_fea'][index]
        E_abs = E_static_fea[:, :, 0]
        E =  E_static_fea[:, :, 1]

        V = self.data['V'][index]
        V_reach_mask = self.data['V_reach_mask'][index]
        V_dispatch_mask = self.data['V_dispatch_mask'][index]

        E_mask = self.data['E_mask'][index]
        route_label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        V_len = self.data['V_len'][index]
        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]
        V_ft = self.data['V_ft'][index]
        td = self.data['t_interval'][index]
        time_label = self.data['time_label'][index]

        return E_abs, E, V, V_reach_mask, V_dispatch_mask, \
                   E_mask, route_label, label_len, V_len, start_fea, start_idx, V_ft, td, time_label
