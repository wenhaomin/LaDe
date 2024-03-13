# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from tqdm import tqdm
from algorithm.cproute_0303.Dataset import CPRouteDataset
from utils.util import  to_device, run, get_nonzeros_nrl, dict_merge

def process_batch(batch, model, device, params):
    def build_loss(outputs, target, pad_value):
        unrolled = outputs.view(-1, outputs.size(-1))
        return F.cross_entropy(unrolled, target.long().view(-1), ignore_index=pad_value)

    batch = to_device(batch, device)
    V, V_len, V_reach_mask, label, label_len, aoi_node_feature = batch
    outputs, pointers = model(V, V_reach_mask, aoi_node_feature)
    loss = build_loss(outputs, label, params['pad_value'])

    return outputs, loss

def test_model(model, test_loader, device, pad_value, params, save2file, mode):
    from utils.eval import Metric
    model.eval()
    evaluators = [Metric([1, 5]),  Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]
    with torch.no_grad():

        for batch in tqdm(test_loader):
            batch = to_device(batch, device)
            V, V_len, V_reach_mask, label, label_len, aoi_node_feature = batch
            outputs, pointers = model(V, V_reach_mask, aoi_node_feature)
            N = outputs.size()[-1]

            pred_steps, label_steps, labels_len, preds_len = get_nonzeros_nrl(pointers.reshape(-1, N), label.reshape(-1, N),
                                                                              label_len.reshape(-1), V_len.reshape(-1), pad_value)

            for e in evaluators:
                e.update(pred_steps, label_steps, labels_len, preds_len)

    if mode == 'val':
        return evaluators[-1]

    for e in evaluators:
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)
    return evaluators[-1]

def main(params):
    params['model'] = 'cproute'
    params['pad_value'] = params['max_task_num'] - 1
    run(params, CPRouteDataset, process_batch, test_model)

def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='deeproute')
    try:
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--sort_x_size', type=int, default=8)
    except:
        pass
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    pass
