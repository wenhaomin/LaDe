# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from pprint import pprint

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.eval import Metric
from utils.util import to_device, ws, dict_merge, get_dataset_path


class BaselineDataset(Dataset):
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
        E_abs_dis = E_static_fea[:, :, 0]
        V_reach_mask = self.data['V_reach_mask'][index]
        time_label = self.data['time_label'][index]
        route_label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        start_idx = self.data['start_idx'][index]
        cou_fea = self.data['cou_fea'][index]

        return E_abs_dis, V_reach_mask, start_idx, cou_fea, route_label, label_len, time_label

def filter_samples(label_steps, label_len, eta_pred, eta_label, pad_value):
    label = []
    label_len_list = []
    eta_pred_list = []
    eta_label_list = []
    for i in range(label_steps.size()[0]):
        if label_steps[i].min().item() != pad_value:
            label.append(label_steps[i].cpu().numpy().tolist())
            label_len_list.append(label_len[i].cpu().numpy().tolist())
            eta_pred_list.append(eta_pred[i].cpu().numpy().tolist())
            eta_label_list.append(eta_label[i].cpu().numpy().tolist())
    return torch.LongTensor(label), \
           torch.LongTensor(label_len_list),\
           torch.LongTensor(eta_pred_list), torch.LongTensor(eta_label_list)

class SpeedETA():
    def __init__(self, params):
        super(SpeedETA, self).__init__()

    def time_prediction(self, distance, start_idx, speed, target_route, length_target):

        B, outputs_route, N = distance.shape[0], [], distance.shape[1]
        outputs_eta = np.zeros([B, N])

        for i in range(B):
            dis, tgt_route, point, tgt_len = distance[i], target_route[i], start_idx[i], length_target[i]
            if tgt_len == 0:
                continue
            else:
                for t in range(int(tgt_len.item())):
                    arrival_time = dis[point][tgt_route[t]] / speed[i]
                    outputs_eta[i][t] = arrival_time

        return outputs_eta


def test_model(model, test_loader, device, params):

    evaluators = [Metric([1, 5]), Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():

        for batch in (test_loader):
            batch = to_device(batch, device)

            E_abs, V_reach_mask, start_idx, cou_fea, route_label, label_len, time_label = batch

            B, T, N = V_reach_mask.size()
            E_abs = torch.repeat_interleave(E_abs.unsqueeze(1), repeats=T, dim=1).reshape(B * T, N, N)

            target = route_label.reshape(B * T, -1)
            speed = torch.repeat_interleave(cou_fea[:, -1].unsqueeze(1), repeats=T, dim=1).reshape(B * T, 1)

            start_idx = start_idx.reshape(-1).long()
            outputs_eta = model.time_prediction(E_abs, start_idx, speed, target, label_len.reshape(-1))

            eta_label = time_label.reshape(B * T, N).to(target.device)
            label, label_len, eta_pred, eta_label = filter_samples(target,  label_len.reshape(-1),  torch.FloatTensor(outputs_eta).to(target.device), eta_label, params['pad_value'])
            for e in evaluators:
                e.update_eta(label_len, eta_pred, eta_label)

    for e in evaluators:
        print(e.eta_to_str())
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'],params_save['eval_max'] = e.len_range
        save2file(params_save)

    return evaluators[-1].eta_to_dict()


def save2file(params):
    from utils.util import save2file_meta
    file_name = ws + f'/output/time_prediction/{params["dataset"]}/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'task', 'eval_min', 'eval_max',
        # model parameters
        'model',
        # training set
        'num_epoch', 'batch_size', 'seed', 'is_test', 'log_time',
        # metric result
       'acc_eta@10','acc_eta@20', 'acc_eta@30', 'acc_eta@40', 'acc_eta@50', 'acc_eta@60', 'mae', 'rmse'
    ]
    save2file_meta(params, file_name, head)

def main(params):
    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    pprint(params)

    test_dataset = BaselineDataset(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    device = torch.device('cpu')

    model = SpeedETA(params)
    result_dict = test_model(model, test_loader, device, params)
    params = dict_merge([result_dict, params])
    return params

def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    import time, nni
    import logging

    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        for data_set in ['delivery_cq_0808']:
            params['dataset'] = data_set
            params.update(tuner_params)
            main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
