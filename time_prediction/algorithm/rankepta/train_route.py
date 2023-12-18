# -*- coding: utf-8 -*-
import torch.nn.functional as F
from tqdm import  tqdm
import torch
from algorithm.ranketpa.dataset import RankEptaDataset
from utils.util import  to_device, run, dict_merge


def process_batch(batch, model, device, params):
    def build_loss(outputs, target, pad_value):
        unrolled = outputs.view(-1, outputs.size(-1))
        return F.cross_entropy(unrolled, target.long().view(-1), ignore_index=pad_value)

    batch = to_device(batch, device)
    V, V_len, V_reach_mask, start_fea, start_idx, route_label, label_len, time_label = batch
    outputs, pointers = model(V, V_reach_mask)
    loss = build_loss(outputs, route_label, params['pad_value'])

    return loss

def get_nonzeros_samples(pred_steps, label_steps, label_len, pad_value):
    pred = []
    label = []
    label_len_list = []

    for i in range(pred_steps.size()[0]):
        #label 不为0时才会考虑该测试该step
        if label_steps[i].min().item() != pad_value:
            label.append(label_steps[i].cpu().numpy().tolist())
            pred.append(pred_steps[i].cpu().numpy().tolist())
            label_len_list.append(label_len[i].cpu().numpy().tolist())

    return torch.LongTensor(pred), torch.LongTensor(label),\
           torch.LongTensor(label_len_list)


def test_model(model, test_dataloader, device, pad_value, params, save2file, mode):
    from utils.eval import Metric
    model.eval()
    evaluators = [Metric([1, 5]), Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():

        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)
            V, V_len, V_reach_mask, start_fea, start_idx, label, label_len, V_at = batch
            outputs, pointers = model(V, V_reach_mask)

            pred_steps, label_steps, labels_len = get_nonzeros_samples(
                pointers.reshape(-1, outputs.size()[-1]), label.reshape(-1, outputs.size()[-1]),
                             label_len.reshape(-1), pad_value)

            for e in evaluators:
                e.update(pred_steps, label_steps, labels_len)

    if mode == 'val':
        return evaluators[-1]

    for e in evaluators:
        print(e.to_str())
        params_save = dict_merge([e.to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)

    return evaluators[-1]

def main(params):
    params['model'] = 'ranketpa_route'
    params['sort_x_size'] = 6
    params['early_stop'] = 6
    params['task'] = 'route'
    run(params, RankEptaDataset, process_batch, test_model)

def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    # Model parameters

    parser.add_argument('--cuda_id', type=int, default=0)
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    params = vars(get_params())
    params['cuda_id'] = 1
    datasets = ['delivery_sh', 'delivery_cq', 'delivery_yt'] # the name of datasets
    args_lst = []
    params['is_test'] = False
    params['early_stop'] = 4
    for hs in [64]:
        for dataset in datasets:
            basic_params = dict_merge([params, {'model': 'ranketpa','dataset': dataset}])
            args_lst.append(basic_params)
    # note: here you can use parallel running to accelerate the experiment.
    for p in args_lst:
        main(p)
