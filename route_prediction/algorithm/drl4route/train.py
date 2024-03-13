# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
from utils.util import *
from algorithm.drl4route.Dataset import DRL4RouteDataset

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from utils.eval import Metric
    model.eval()

    evaluators = [Metric([1, 5]),  Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():

        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_reach_mask, label, label_len = batch
            outputs, pointers, _ = model(V, V_reach_mask, sample = False, type = 'mle')
            N = outputs.size()[-1]
            pred_steps, label_steps, labels_len, preds_len = get_nonzeros_nrl(pointers.reshape(-1, N),
                                                                              label.reshape(-1, N),
                                                                              label_len.reshape(-1), label_len.reshape(-1),
                                                                              pad_value)

            for e in evaluators:
                e.update(pred_steps, label_steps, labels_len, preds_len)

    if mode == 'val':
        return evaluators[-1]

    for e in evaluators:
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)
    return evaluators[-1]

def process_batch(batch, model, device, params):
    batch = to_device(batch, device)
    V, V_reach_mask, label, label_len = batch

    pred_scores, pred_pointers, values = model(V, V_reach_mask, sample=False, type='mle')
    unrolled = pred_scores.view(-1, pred_scores.size(-1))
    N = pred_pointers.size(-1)
    mle_loss = F.cross_entropy(unrolled, label.view(-1), ignore_index=params['pad_value'])
    rl_log_probs, sample_out, sample_values = model(V, V_reach_mask, sample=True, type='rl')
    with torch.autograd.no_grad():
        _, greedy_out, _ = model(V, V_reach_mask, sample=False, type='rl')

    seq_pred_len = torch.sum((pred_pointers.reshape(-1, N) < N - 1) + 0, dim=1)

    sample_out_samples, greedy_out_samples, label_samples, label_len_samples, rl_log_probs_samples, seq_pred_len_samples = \
        get_reinforce_samples(sample_out.reshape(-1, N), greedy_out.reshape(-1, N), label.reshape(-1, N), label_len.reshape(-1), params['pad_value'], rl_log_probs, seq_pred_len)

    log_prob_mask = get_log_prob_mask(seq_pred_len_samples, params)

    rl_log_probs_samples = rl_log_probs_samples * log_prob_mask

    rl_log_probs_samples = torch.sum(rl_log_probs_samples, dim=1) / seq_pred_len_samples

    krc_reward, lsd_reward, acc_3_reward = calc_reinforce_rewards(sample_out_samples, label_samples, label_len_samples, params)

    baseline_krc_reward, baseline_lsd_reward, baseline_acc_3_reward = calc_reinforce_rewards(greedy_out_samples, label_samples, label_len_samples, params)

    reinforce_loss = -torch.mean(torch.tensor(baseline_lsd_reward - lsd_reward).to(rl_log_probs_samples.device) * rl_log_probs_samples)

    loss = mle_loss + params['rl_ratio'] * reinforce_loss


    return pred_pointers, loss

def main(params):
    params['pad_value'] = params['max_task_num'] - 1
    run(params, DRL4RouteDataset, process_batch, test_model)

def get_params():
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args
