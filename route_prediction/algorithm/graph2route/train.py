# -*- coding: utf-8 -*-

import numpy as np
from tqdm import  tqdm

import torch
import torch.nn.functional as F

from utils.eval import Metric
from utils.util import get_nonzeros_nrl, run, dict_merge
from algorithm.graph2route.Graph2Route import Graph2RouteDataset

def collate_fn(batch):
    return  batch

def batch2input(batch, device):
    V, V_len, V_reach_mask, E_static_fea, E_mask, A, start_fea, start_idx, cou_fea, label, label_len = zip(*batch)

    V = torch.FloatTensor(V).to(device)
    B, T, N, _ = V.shape
    E_mask = np.array(E_mask)
    A = np.array(A)
    E_static_fea = np.array(E_static_fea)
    E = torch.zeros([B, T, N, N, 4]).to(device)  # Edge feature, E: (B, T, N, N, d_e)
    for t in range(T):
        # A_t = np.expand_dims(A[:, t, :, :], axis=-1)
        # E_t = torch.Tensor(np.concatenate([E_static_fea, A_t], axis=3)).to(device)  # E_t: (B, N, N, 5)
        E_t = torch.Tensor(E_static_fea).to(device)  # E_t: (B, N, N, 4)
        E_mask_t = torch.Tensor(E_mask[:, t, :, :]).to(device)  # (B, N, N)
        E_t = E_t * E_mask_t.unsqueeze(-1).expand(E_t.shape)
        E[:, t, :, :, :] = E_t

    V_reach_mask = torch.BoolTensor(V_reach_mask).to(device)
    label = torch.LongTensor(np.array(label)).to(device)
    label_len = torch.LongTensor(label_len).to(device)
    start_fea = torch.FloatTensor(start_fea).to(device)
    start_idx = torch.LongTensor(start_idx).to(device)
    cou_fea = torch.LongTensor(cou_fea).to(device)
    return  V, V_reach_mask, E, start_fea, start_idx, cou_fea, label, label_len

def process_batch_weighted(batch, model, device, pad_vaule):
    V, V_reach_mask, E, start_fea, cou_fea, start_idx, label, label_len = batch2input(batch, device)
    pred_scores, pred_pointers = model(V, V_reach_mask, E, start_fea, start_idx, cou_fea )
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=pad_vaule)
    return pred_pointers, loss

def process_batch(batch, model, device, params):
    V, V_reach_mask,  E, start_fea,  start_idx, cou_fea, label, label_len = batch2input(batch, device)

    pred_scores, pred_pointers = model(V, V_reach_mask, E, start_fea, start_idx, cou_fea)
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    loss = F.cross_entropy(unrolled, label.view(-1), ignore_index = params['pad_value'])
    return pred_pointers, loss

def test_model(model, test_loader, device, pad_value, params, save2file, mode):
    model.eval()
    evaluators = [Metric([1, 5]),  Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():
        for batch in tqdm(test_loader):
            V, V_reach_mask,  E, start_fea, start_idx, cou_fea, label, label_len = batch2input(batch, device)
            pred_scores, pred_pointers = model(V, V_reach_mask, E, start_fea, start_idx, cou_fea)

            N = pred_pointers.size(-1)
            pred_len = torch.sum((pred_pointers.reshape(-1, N) < N - 1) + 0, dim=1)

            pred_steps, label_steps, labels_len, preds_len = get_nonzeros_nrl(pred_pointers.reshape(-1, N), label.reshape(-1, N),
                             label_len.reshape(-1), pred_len, pad_value)

            for e in evaluators:
                e.update(pred_steps, label_steps, labels_len, preds_len)

        evaluator  = evaluators[-1]
        if mode == 'val':
            return evaluator

        for e in evaluators:
            params_save = dict_merge([e.eta_to_dict(), params])
            params_save['eval_min'], params_save['eval_max'] = e.len_range
            save2file(params_save)
        return evaluator

def main(params):
    params['model'] = 'graph2route'
    params['pad_value'] = params['max_task_num'] - 1
    params['long_loss_weight'] = 1.5
    # params['gcn_num_layers'] = 3

    run(params, Graph2RouteDataset, process_batch, test_model, collate_fn)

def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":

    import time, nni
    import logging
    from Graph2Route import Graph2Route_pickup, save2file

    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        # deeproute: 128
        params['dataset'] = 'pickup_yt_0614_dataset_change'
        params['batch_size'] = 32
        params['worker_emb_dim'] = 20
        params['hidden_size'] = 64
        params['gcn_num_layers'] = 3
        params['model_save2file'] = (Graph2Route_pickup, save2file)

        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise