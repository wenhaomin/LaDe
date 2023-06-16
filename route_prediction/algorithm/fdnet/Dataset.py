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
        path_key = {'train':'train_path', 'val':'test_path','test':'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['V_len'])

    def __getitem__(self, index):
        E_static_fea = self.data['E_static_fea'][index]
        E_abs_dis = E_static_fea[:, :, 0]
        E_dis =  E_static_fea[:, :, 1]
        E_mask = self.data['E_mask'][index]

        V = self.data['V'][index]
        V_pt = self.data['V_pt'][index]
        V_ft = self.data['V_ft'][index]
        V_len = self.data['V_len'][index]
        V_reach_mask = self.data['V_reach_mask'][index]
        V_dispatch_mask = self.data['V_dispatch_mask'][index]

        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]

        label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]

        td = self.data['t_interval'][index]

        return E_abs_dis, E_dis, V, V_reach_mask, V_dispatch_mask, \
                   E_mask, label, label_len, V_len, start_fea, start_idx, V_pt, V_ft, td


if __name__ == '__main__':
    pass


