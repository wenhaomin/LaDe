import numpy as np
from torch.utils.data import Dataset


class LGBDataset(Dataset):
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
        self.max_iter = len(self.data['label_len'])

    def __len__(self):
        return len(self.data['label_len'])

    def __getitem__(self, index):

        V = self.data['V'][index]
        V_reach_mask = self.data['V_reach_mask'][index]

        route_label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        start_fea = self.data['start_fea'][index]
        time_label = self.data['time_label'][index]

        return  V, V_reach_mask, start_fea, route_label, label_len, time_label
