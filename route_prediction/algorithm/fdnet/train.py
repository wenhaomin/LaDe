# -*- coding: utf-8 -*-
from tqdm import tqdm
from pprint import pprint

import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.util import *
import algorithm.fdnet.FDNet as FDNet
from algorithm.fdnet.Dataset import FDNetDataset

criterion_tp = nn.L1Loss()

def build_loss(outputs, target, pad_value):
    unrolled = outputs.view(-1, outputs.size(-1))
    return F.cross_entropy(unrolled, target.view(-1).long(), ignore_index=pad_value)

def train_val_test_fd(train_loader, val_loader, test_loader, model_rp, model_tp, device, params, save2file):

    print('current device:', device)

    model_rp.to(device)
    model_tp.to(device)
    optimizer_rp = torch.optim.Adam(model_rp.parameters(), lr=params['lr'], weight_decay=params['wd'])
    optimizer_tp = torch.optim.Adam(model_tp.parameters(), lr=params['lr_tp'], weight_decay=params['wd'])
    early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])
    model_name = model_rp.model_file_name()
    model_tp_name = model_name + '-tp'
    model_path = ws + f'/data/dataset/{params["dataset"]}/sort_model/{model_name}'
    model_tp_path = ws + f'/data/dataset/{params["dataset"]}/sort_model/{model_tp_name}'
    dir_check(model_path)
    for epoch in range(params['num_epoch']):

        if early_stop.stop_flag: break
        postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            ave_loss = None
            model_rp.train()
            model_tp.train()
            for i, batch in enumerate(t):
                batch = to_device(batch, device)
                E_abs, E, V, V_reach_mask, V_dispatch_mask, \
                E_mask, label, label_len, V_len, start_fea, start_idx, V_pt, V_ft, td = batch

                outputs, pointers, time_duration_predict = model_rp(V, E, V_pt, V_ft,
                                                                    label_len, V_reach_mask, label, model_tp,
                                                                    start_idx, E_abs,
                                                                    E_mask, V_dispatch_mask, start_fea,
                                                                    mode='teacher_force')

                loss_rp = build_loss(outputs, label, params['pad_value'])
                loss_tp = criterion_tp(time_duration_predict, td.reshape(-1, label.size(-1)))
                if ave_loss is None:
                    ave_loss = loss_rp.item()
                else:
                    ave_loss = ave_loss * i / (i + 1) + loss_rp.item() / (i + 1)
                postfix["loss"] = ave_loss
                postfix["current_loss"] = loss_rp.item()
                t.set_postfix(**postfix)

                optimizer_tp.zero_grad()
                optimizer_rp.zero_grad()
                loss_rp.backward(retain_graph = True)
                loss_tp.backward()
                optimizer_rp.step()
                optimizer_tp.step()

        val_result = test_model(model_rp, model_tp, val_loader, device, params, save2file, mode = 'val')
        print('\nval result:', val_result.to_str(), 'Best krc:', round(early_stop.best_metric(),3), '| Best epoch:', early_stop.best_epoch)
        is_best_change = early_stop.append(val_result.to_dict()['krc'])
        if is_best_change: 
            torch.save(model_rp.state_dict(), model_path)
            torch.save(model_tp.state_dict(), model_tp_path)
        if params['is_test']:
            torch.save(model_rp.state_dict(), model_path)
            torch.save(model_tp.state_dict(), model_tp_path)
            break
        nni.report_intermediate_result(val_result.to_dict()['krc'])

    try:
        model_rp.load_state_dict(torch.load(model_path))
        model_tp.load_state_dict(torch.load(model_tp_path))
        print('best model_rp path', model_path)
        print('best model_tp path', model_tp_path)
        print('best model loaded!!')
    except:
        print('load best model failed')
    test_result = test_model(model_rp, model_tp, test_loader, device, params, save2file, mode = 'test') #
    print('\n-------------------------------------------------------------')
    print('Evaluation in test:', test_result.to_str())

    nni.report_final_result(test_result.to_dict())
    return test_result.to_dict()

def test_model(model_rp, model_tp, test_dataloader, device, params,save2file, mode):
    from utils.eval import Metric
    model_rp.eval()
    model_tp.eval()
    evaluators = [Metric([1, 5]),  Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)

            E_abs_dis, E_dis, V, V_reach_mask, V_dispatch_mask, \
            E_mask, label, label_len, V_len, start_fea, start_idx,  V_pt, V_ft, td= batch

            pointers = model_rp(V, E_dis, V_pt, V_ft, label_len, V_reach_mask, label, model_tp,
                                start_idx, E_abs_dis, E_mask, V_dispatch_mask, start_fea, mode=params['prediction_method'])

            pred_steps, label_steps, labels_len, preds_len = \
                get_nonzeros_nrl(pointers.reshape(-1, label.size(-1)), label.reshape(-1, label.size(-1)), label_len.reshape(-1), V_len.reshape(-1), params['pad_value'])

        for e in evaluators:
            e.update(pred_steps, label_steps, labels_len, preds_len)

    evaluator  = evaluators[-1]
    if mode == 'val':
        return evaluator

    else:
        for e in evaluators:
            params_save = dict_merge([e.eta_to_dict(), params])
            params_save['eval_min'], params_save['eval_max'] = e.len_range
            save2file(params_save)
        return evaluator


def main(params):
    device = torch.device(f'cuda:{params["cuda_id"]}' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    pprint(params) 

    train_dataset = FDNetDataset(mode='test', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True) 

    val_dataset = FDNetDataset(mode='test', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)  

    test_dataset = FDNetDataset(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    model_rp, model_tp, save2file = FDNet.FDNet, FDNet.TimePrediction, FDNet.save2file
    model_rp = model_rp(params)
    model_tp = model_tp(params)
    result_dict = train_val_test_fd(train_loader, val_loader, test_loader, model_rp, model_tp, device, params, save2file)
    params = dict_merge([result_dict, params])

    return params

def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    parser.add_argument('--prediction_method', type=str, default='beam_search')
    parser.add_argument('--lr_rp', type=float, default=1e-3)
    parser.add_argument('--lr_tp', type=float, default=1e-3)
    parser.add_argument('--model', type=str, default='fdnet_logistics')
    parser.add_argument('--num_nodes', type=int, default=25)
    parser.add_argument('--time_dif_dim', type=int, default=1, help='dim of time dif')
    parser.add_argument('--dis_dif_dim', type=int, default=2, help='dim of dis dif')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":

    import time
    import nni
    import logging
    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
