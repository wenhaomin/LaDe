# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from utils.util import *
from utils.eval import Metric
from algorithm.fdnet.Dataset import FDNetDataset
import algorithm.fdnet.FDNet as FDNet
criterion_tp = nn.L1Loss()

def get_nonzeros_eta(pred_steps, label_steps, label_len, eta_pred, eta_label, pad_value):
    pred = []
    label = []
    label_len_list = []
    eta_pred_list = []
    eta_label_list = []

    for i in range(pred_steps.size()[0]):
        if label_steps[i].min().item() != pad_value:
            label.append(label_steps[i].cpu().numpy().tolist())
            pred.append(pred_steps[i].cpu().numpy().tolist())
            label_len_list.append(label_len[i].cpu().numpy().tolist())
            eta_pred_list.append(eta_pred[i].cpu().numpy().tolist())
            eta_label_list.append(eta_label[i].cpu().numpy().tolist())
    return torch.LongTensor(pred), torch.LongTensor(label), \
           torch.LongTensor(label_len_list), torch.LongTensor(eta_pred_list), torch.LongTensor(eta_label_list)

def build_loss(outputs, target, pad_value):
    unrolled = outputs.view(-1, outputs.size(-1))
    return F.cross_entropy(unrolled, target.view(-1).long(), ignore_index=pad_value)

def train_val_test_fd(train_loader, val_loader, test_loader, model_rp, model_tp, device, params, save2file):

    print('current device:', device)

    model_rp.to(device)
    model_tp.to(device)
    optimizer_rp = torch.optim.Adam(model_rp.parameters(), lr=params['lr'], weight_decay=params['wd'])
    optimizer_tp = torch.optim.Adam(model_tp.parameters(), lr=params['lr'], weight_decay=params['wd'])
    early_stop = EarlyStop(mode='minimize', patience=params['early_stop'])
    model_name = model_rp.model_file_name()
    model_tp_name = model_name + '-tp'
    model_path = ws + f'/data/dataset/{params["dataset"]}/{params["model"]}/{model_name}'
    model_tp_path = ws + f'/data/dataset/{params["dataset"]}/{params["model"]}/{model_tp_name}'

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
                E_mask, label, label_len, V_len, start_fea, start_idx, V_ft, td, V_at = batch

                outputs, pointers, time_duration_predict, eta_predict, t_mask = model_rp(V, E, V_ft,
                                                                    label_len, V_reach_mask, model_tp,
                                                                    start_idx, E_abs,
                                                                    E_mask, V_dispatch_mask, start_fea,
                                                                    mode='teacher_force')

                loss_rp = build_loss(outputs, label, params['pad_value'])
                loss_tp = criterion_tp(eta_predict, V_at.reshape(-1, label.size(-1)) * t_mask)

                if ave_loss is None:
                    ave_loss = loss_rp.item()
                else:
                    ave_loss = ave_loss * i / (i + 1) + loss_rp.item() / (i + 1)
                postfix["loss"] = ave_loss
                postfix["current_loss"] = loss_rp.item()
                t.set_postfix(**postfix)

                optimizer_tp.zero_grad()
                optimizer_rp.zero_grad()

                loss_tp.backward(retain_graph=True)
                loss_rp.backward()

                optimizer_tp.step()
                optimizer_rp.step()

        val_result = test_model(model_rp, model_tp, val_loader, device, params, save2file, mode = 'val')
        print('\nval result:', val_result.eta_to_str(), 'Best mae:', round(early_stop.best_metric(),3), '| Best epoch:', early_stop.best_epoch)
        is_best_change = early_stop.append(val_result.eta_to_dict()['mae'])
        if is_best_change:
            torch.save(model_rp.state_dict(), model_path)
            torch.save(model_tp.state_dict(), model_tp_path)
            print('\nval result-best saved:', val_result.eta_to_str())
        if params['is_test']:
            torch.save(model_rp.state_dict(), model_path)
            torch.save(model_tp.state_dict(), model_tp_path)
            break
    try:
        model_rp.load_state_dict(torch.load(model_path))
        model_tp.load_state_dict(torch.load(model_tp_path))
        print('best model path', model_path)
        print('best model loaded!!')
    except:
        print('load best model failed')
    test_result = test_model(model_rp, model_tp, test_loader, device, params, save2file, mode = 'test')
    print('\n-------------------------------------------------------------')
    print('Best epoch: ', early_stop.best_epoch)
    print('Evaluation in test:', test_result.eta_to_str())

    return test_result.eta_to_dict()

def test_model(model_rp, model_tp, test_dataloader, device, params,save2file, mode):
    model_rp.eval()
    model_tp.eval()
    evaluators = [Metric([1, 5]),  Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = to_device(batch, device)

            E_abs, E, V, V_reach_mask, V_dispatch_mask, \
            E_mask, label, label_len, V_len, start_fea, start_idx, V_ft, td, V_at = batch

            pointers, eta_pred = model_rp(V, E, V_ft, label_len, V_reach_mask, model_tp, start_idx,  E_abs, E_mask, V_dispatch_mask, start_fea, mode=params['prediction_method'])

            pred_steps, label_steps, labels_len, eta_preds, eta_labels =  get_nonzeros_eta(pointers.reshape(-1, label.size(-1)), label.reshape(-1, label.size(-1)), label_len.reshape(-1), eta_pred.reshape(-1, label.size(-1)), V_at.reshape(-1, label.size(-1)), params['pad_value'])
            for e in evaluators:
                e.update_eta(labels_len, eta_preds, eta_labels)

    if mode == 'val':
        return evaluators[-1]
    for e in evaluators:
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)
    return evaluators[-1]

def main(params):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['prediction_method'] = 'greedy'
    params['device'] = device

    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    pprint(params)

    train_dataset = FDNetDataset(mode='train', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

    val_dataset = FDNetDataset(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True)

    test_dataset = FDNetDataset(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True)

    model_rp, model_tp, save2file = FDNet.FDNet, FDNet.TimePrediction, FDNet.save2file
    model_rp = model_rp(params)
    model_tp = model_tp(params)

    result_dict = train_val_test_fd(train_loader, val_loader, test_loader, model_rp, model_tp, device, params, save2file)
    params = dict_merge([result_dict, params])

    return params

def get_params():
    from utils.util import get_common_params
    parser = get_common_params()

    args, _ = parser.parse_known_args()
    return args
