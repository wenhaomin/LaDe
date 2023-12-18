# -*- coding: utf-8 -*-

import torch.nn.functional as F
from tqdm import  tqdm
import torch
from utils.util import to_device, dict_merge
from algorithm.mlp.Dataset import MLPDataset

def get_eta_result(pred, label, label_len, label_route):
    N = label_route.shape[2]
    B = label_route.shape[0]
    T = label_route.shape[1]
    pred_result = torch.zeros(B*T, N).to(label.device)
    label_result = torch.zeros(B*T, N).to(label.device)
    label_len = label_len.reshape(B*T)
    label = label.reshape(B*T, N)

    label_len_list = []
    eta_pred_list = []
    eta_label_list = []

    label_route = label_route.reshape(B * T, N)
    for i in range(B*T):
        if label_len[i].long().item() != 0:
            pred_result[i][:label_len[i].long().item()] = pred[i][label_route[i][:label_len[i].long().item()]]
            label_result[i][:label_len[i].long().item()] = label[i][:label_len[i].long().item()]

    for i in range(B*T):
        if label_len[i].long().item() != 0:
            eta_label_list.append(label_result[i].detach().cpu().numpy().tolist())
            eta_pred_list.append(pred_result[i].detach().cpu().numpy().tolist())
            label_len_list.append(label_len[i].detach().cpu().numpy().tolist())

    return  torch.LongTensor(label_len_list), torch.LongTensor(eta_pred_list), torch.LongTensor(eta_label_list)

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from utils.eval import Metric
    model.eval()
    evaluators = [Metric([1, 5]), Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():

        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_reach_mask, route_label, label_len, time_label, start_fea, cou_fea = batch
            eta = model(V, V_reach_mask, start_fea, cou_fea)
            label_len, eta_pred, eta_label = get_eta_result(eta, time_label, label_len, route_label)

            for e in evaluators:
                e.update_eta(label_len, eta_pred, eta_label)
    evaluator = evaluators[-1]
    if mode == 'val':
        return evaluator
    else:
        for e in evaluators:
            params_save = dict_merge([e.eta_to_dict(), params])
            params_save['eval_min'],params_save['eval_max'] = e.len_range
            save2file(params_save)
        return evaluators[-1]


def eta_mae_loss_calc(V_at, label_len, eta, label_route):
    N = eta.shape[1]
    B = V_at.shape[0]
    T = V_at.shape[1]
    V_at = V_at.reshape(B*T, N)
    label_route = label_route.reshape(B * T, N)
    label_len = label_len.reshape(B * T, 1)
    pred_result = torch.empty(0).to(V_at.device)
    label_result = torch.empty(0).to(V_at.device)
    for i in range(len(eta)):
        pred_result = torch.cat([pred_result, eta[i][label_route[i][:label_len[i].long().item()]]])
        label_result = torch.cat([label_result, V_at[i][:label_len[i].long().item()]])
    return F.l1_loss(pred_result, label_result)


def process_batch(batch, model, device, params):
    batch = to_device(batch, device)
    V, V_reach_mask, route_label, label_len, time_label, start_fea, cou_fea = batch

    eta = model(V, V_reach_mask, start_fea, cou_fea)

    eta_loss = eta_mae_loss_calc(time_label, label_len, eta, route_label)

    return eta_loss

def main(params):
    params['model'] = 'mlp'
    params['pad_value'] = params['max_task_num'] - 1
    params['task'] = 'time'
    from utils.util import run
    run(params, MLPDataset, process_batch, test_model)


def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='mlp')
    # parser.add_argument('--hidden_size', type=int, default=128)
    args, _ = parser.parse_known_args()
    return args
