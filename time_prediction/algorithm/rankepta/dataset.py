from torch.utils.data import Dataset
import numpy as np

def merge_first_two_dim(arr):
    # get the dimension of the arr
    if arr.ndim < 2:
        raise ValueError("not enough dimension")
    # merge the first two dimension
    merged_arr = arr.reshape(-1, *arr.shape[2:])
    return merged_arr

class ModelDataset(Dataset):
    def __init__(self, V, V_reach_mask, label, label_len, V_at, sort_idx,  sample_num):
        super(ModelDataset, self).__init__()
        self.V = V
        self.V_reach_mask = V_reach_mask
        self.label = label
        self.label_len = label_len
        self.V_at = V_at
        self.sort_idx = sort_idx
        self.sample_num = sample_num

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        V = self.V[index]
        V_reach_mask = self.V_reach_mask[index]
        label = self.label[index]
        label_len = self.label_len[index]
        V_at = self.V_at[index]
        sort_idx = self.sort_idx[index]
        return V, V_reach_mask, label, label_len, V_at, sort_idx

        # return V, V_reach_mask, label, label_len, V_at, sort_idx, route_label_all, eta_label_len
    
class RankEptaDataset(Dataset):
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
        V_len = self.data['label_len'][index]
        V_reach_mask = self.data['V_reach_mask'][index]

        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]

        route_label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        time_label = self.data['time_label'][index]



        return  V, V_len, V_reach_mask, start_fea, start_idx, route_label, label_len, time_label
