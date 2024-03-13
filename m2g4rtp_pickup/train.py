# -*- coding: utf-8 -*-
import os
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from tqdm import  tqdm
import torch
from utils.util import to_device, dict_merge
from algorithm.m2g4rtp_pickup.Dataset import M2GDataset
import torch.nn as nn

class uncertainty_loss(nn.Module):
    def __init__(self, num_task):
        super(uncertainty_loss, self).__init__()
        sigma = torch.randn(num_task)
        self.sigma = nn.Parameter(sigma)
        self.num_task = num_task

    def forward(self, *inputs):
        total_loss = 0
        index = 0
        for loss in inputs:
            if loss == 0:
                continue
            sigma_sq = torch.pow(self.sigma[index], 2)
            total_loss += 0.5 * torch.pow(torch.exp(sigma_sq), -1) * loss + sigma_sq
            index += 1
        assert index == self.num_task
        return total_loss

def get_eta_result(pred, label, label_len, label_route,  pred_pointers):
    N = label_route.shape[1]
    B = label_route.shape[0]
    label_len = label_len.reshape(B)
    label = label.reshape(B, N)

    label_len_list = []
    eta_pred_list = []
    eta_label_list = []
    route_pred_list = []
    route_label_list = []

    label_route = label_route.reshape(B, N)

    for i in range(B):
        if label_len[i].long().item() != 0:
            eta_label_list.append(label[i].detach().cpu().numpy().tolist())
            eta_pred_list.append(pred[i].detach().cpu().numpy().tolist())
            label_len_list.append(label_len[i].detach().cpu().numpy().tolist())
            route_pred_list.append(pred_pointers[i].detach().cpu().numpy().tolist())
            route_label_list.append(label_route[i].detach().cpu().numpy().tolist())

    return  torch.LongTensor(label_len_list), torch.LongTensor(eta_pred_list), torch.LongTensor(eta_label_list), torch.LongTensor(route_pred_list), torch.LongTensor(route_label_list)

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from utils.eval import Metric
    model.eval()
    evaluators = [Metric([1, 5]),  Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():

        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_reach_mask, label, label_len, V_at, start_fea, cou_fea, \
            aoi_feature_steps, aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_idx_steps, aoi_index_steps, E_static_fea, E_mask = batch
            B, T, N, _ = E_mask.shape
            E = torch.zeros([B, T, N, N, 4]).to(V.device)

            for t in range(T):
                E[:, t, :, :, :] = E_static_fea * E_mask[:, t, :, :].unsqueeze(-1).expand(E_static_fea.shape)
            V, V_reach_mask, cou_fea, label, label_len, V_at, start_fea, aoi_feature_steps, \
            aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_index_steps, aoi_idx_steps, E = \
                filter_input(V, V_reach_mask, cou_fea, label, label_len, V_at, start_fea,
                             aoi_feature_steps, aoi_start_steps, aoi_pos_steps, aoi_len_steps,
                             aoi_index_steps, aoi_idx_steps, E)

            pred_scores, pred_pointers, eta, aoi_order_score = model(V, V_reach_mask, start_fea, cou_fea,
                                                                     aoi_feature_steps, aoi_start_steps,
                                                                     aoi_pos_steps, aoi_len_steps, aoi_index_steps, E,
                                                                     is_train=False)

            label_len, eta_pred, eta_label, route_pred, route_label = get_eta_result(eta, V_at, label_len, label, pred_pointers)
            for e in evaluators:
                e.update_route_eta(route_pred, route_label, label_len, eta_pred, eta_label)

    if mode == 'val':
        print(evaluators[-1].route_eta_to_str())
        return evaluators[-1]

    for e in evaluators:
        params_save = dict_merge([e.route_eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)
    return evaluators[-1]

def eta_mae_loss_calc(V_at, label_len, eta):
    N = eta.shape[1]
    B = V_at.shape[0]
    V_at = V_at.reshape(B, N)
    label_len = label_len.reshape(B, 1)
    pred_result = torch.empty(0).to(V_at.device)
    label_result = torch.empty(0).to(V_at.device)
    for i in range(len(label_len)):
        lab_len = label_len[i]
        lab = V_at[i][:lab_len.long()]
        pred = eta[i][:lab_len.long()]
        pred_result = torch.cat([pred_result, pred])
        label_result = torch.cat([label_result, lab])
    return F.l1_loss(pred_result, label_result)

def filter_input(V, V_reach_mask, cou_fea, label, label_len, V_at, start_fea,
                     aoi_feature_steps,  aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_index_steps, aoi_idx_steps, E):
    B, T, N = V_reach_mask.shape[0], V_reach_mask.shape[1], V_reach_mask.shape[2]
    valid_index = (label_len.reshape(B*T) != 0).nonzero().squeeze(1)
    V = V.reshape(B * T, N, -1)[valid_index]
    cou_fea = cou_fea.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)[valid_index]
    label = label.reshape(B * T, N)[valid_index]
    label_len = label_len.reshape(B * T)[valid_index]
    V_at = V_at.reshape(B * T, N)[valid_index]
    start_fea = start_fea.reshape(B * T, -1)[valid_index]
    V_reach_mask = V_reach_mask.reshape(B * T, N)[valid_index]

    aoi_feature_steps = aoi_feature_steps.reshape(B*T, 10, -1)[valid_index]
    aoi_start_steps = aoi_start_steps.reshape(B*T, -1)[valid_index]
    aoi_pos_steps = aoi_pos_steps.reshape(B*T, -1)[valid_index]
    aoi_len_steps = aoi_len_steps.reshape(B*T)[valid_index]
    aoi_index_steps = aoi_index_steps.reshape(B*T, -1)[valid_index]
    aoi_idx_steps = aoi_idx_steps.reshape(B*T, -1)[valid_index]
    E = E.reshape(B * T, N, N, -1)[valid_index]

    return V, V_reach_mask, cou_fea, label, label_len, V_at, start_fea, aoi_feature_steps,\
           aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_index_steps, aoi_idx_steps, E

def process_batch(batch, model, device, params):
    class uncertainty_loss(nn.Module):
        def __init__(self, num_task):
            super(uncertainty_loss, self).__init__()
            sigma = torch.randn(num_task)
            self.sigma = nn.Parameter(sigma)
            self.num_task = num_task

        def forward(self, *inputs):
            total_loss = 0
            index = 0
            for loss in inputs:
                if loss == 0:
                    continue
                sigma_sq = torch.pow(self.sigma[index], 2)
                total_loss += 0.5 * torch.pow(torch.exp(sigma_sq), -1) * loss + sigma_sq
                index += 1
            assert index == self.num_task
            return total_loss

    batch = to_device(batch, device)
    V, V_reach_mask, label, label_len, V_at, start_fea, cou_fea, \
    aoi_feature_steps, aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_idx_steps, aoi_index_steps, E_static_fea, E_mask = batch
    B, T, N, _ = E_mask.shape
    E = torch.zeros([B, T, N, N, 4]).to(V.device)

    for t in range(T):
        E[:, t, :, :, :] = E_static_fea * E_mask[:, t, :, :].unsqueeze(-1).expand(E_static_fea.shape)
    V, V_reach_mask, cou_fea, label, label_len, V_at, start_fea, aoi_feature_steps, \
    aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_index_steps, aoi_idx_steps, E = \
        filter_input(V, V_reach_mask, cou_fea, label, label_len, V_at, start_fea,
                     aoi_feature_steps,  aoi_start_steps, aoi_pos_steps, aoi_len_steps,
                     aoi_index_steps, aoi_idx_steps, E)

    pred_scores, pred_pointers, eta, aoi_order_score = model(V, V_reach_mask, start_fea, cou_fea,
                                                             aoi_feature_steps, aoi_start_steps,
                                                             aoi_pos_steps, aoi_len_steps, aoi_index_steps, E, is_train=True)

    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    unrolled_aoi = aoi_order_score.view(-1, aoi_order_score.size(-1))
    aoi_idx_steps[aoi_idx_steps == -1] = aoi_order_score.size(-1) - 1
    mle_loss_aoi = F.cross_entropy(unrolled_aoi, aoi_idx_steps.view(-1), ignore_index=params['pad_value'])
    mle_loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=params['pad_value'])
    eta_loss = eta_mae_loss_calc(V_at, label_len, eta)
    loss = mle_loss + mle_loss_aoi + eta_loss
    return (pred_pointers, eta), loss

def main(params):
    params['sort_x_size'] = 10
    params['pad_value'] = params['max_task_num'] - 1
    params['start_fea'] = 5
    from utils.util import run

    run(params, M2GDataset, process_batch, test_model)

if __name__ == '__main__':
    pass
