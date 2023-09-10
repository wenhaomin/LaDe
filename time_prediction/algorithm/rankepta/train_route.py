# -*- coding: utf-8 -*-
import os
import argparse

import torch.nn.functional as F
from tqdm import  tqdm
import torch
from algorithm.ranketpa.dataset import RankEptaDataset
from utils.util import  to_device, run, get_nonzeros_nrl, dict_merge


def process_batch(batch, model, device, params):
    def build_loss(outputs, target, pad_value):
        unrolled = outputs.view(-1, outputs.size(-1))
        return F.cross_entropy(unrolled, target.long().view(-1), ignore_index=pad_value)

    batch = to_device(batch, device)
    V, V_len, V_reach_mask, start_fea, start_idx, route_label, label_len, time_label = batch
    outputs, pointers = model(V, V_reach_mask)
    loss = build_loss(outputs, route_label, params['pad_value'])

    return outputs, loss

def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from utils.eval import Metric
    model.eval()
    evaluators = [Metric([1, 5]), Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():

        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_len, V_reach_mask, start_fea, start_idx, label, label_len, V_at = batch
            outputs, pointers = model(V, V_reach_mask)

            pred_steps, label_steps, labels_len, preds_len = \
                get_nonzeros_nrl(pointers.reshape(-1, outputs.size()[-1]), label.reshape(-1, outputs.size()[-1]),
                             label_len.reshape(-1), V_len.reshape(-1), pad_value)

            for e in evaluators:
                e.update(label_len, pred_steps, label_steps)

    for e in evaluators:
        print(e.to_str())
        params_save = dict_merge([e.to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)

    return evaluators[-1].to_dict()

def main(params):
    params['model'] = 'ranketpa'
    params['sort_x_size'] = 8
    params['early_stop'] = 6
    run(params, RankEptaDataset, process_batch, test_model)

def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='deeproute_logistics')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--sort_x_size', type=int, default=8)
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    import time
    import logging

    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('GPU:', torch.cuda.current_device())
    try:
        params = vars(get_params())

        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
