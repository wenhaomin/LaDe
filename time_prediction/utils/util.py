# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from tqdm import  tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file
ws =  get_workspace()

def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

def whether_stop(metric_lst = [], n=2, mode='maximize'):
    '''
    For fast parameter search, judge wether to stop the training process according to metric score
    n: Stop training for n consecutive times without rising
    mode: maximize / minimize
    '''
    if len(metric_lst) < 1:return False # at least have 2 results.
    if mode == 'minimize': metric_lst = [-x for x in metric_lst]
    max_v = max(metric_lst)
    max_idx = 0
    for idx,v in enumerate(metric_lst):
        if v == max_v:max_idx = idx
    return max_idx < len(metric_lst) - n

from multiprocessing import Pool
def multi_thread_work(parameter_queue,function_name,thread_number=5):
    pool = Pool(thread_number)
    result = pool.map(function_name, parameter_queue)
    pool.close()
    pool.join()
    return  result

class EarlyStop():
    """
    For training process, early stop strategy
    """
    def __init__(self, mode='maximize', patience = 1):
        self.mode = mode
        self.patience =  patience
        self.metric_lst = []
        self.stop_flag = False
        self.best_epoch = -1 # the best epoch
        self.is_best_change = False # whether the best change compare to the last epoch

    def append(self, x):
        self.metric_lst.append(x)
        #update the stop flag
        self.stop_flag = whether_stop(self.metric_lst, self.patience, self.mode)
        #update the best epoch
        best_epoch = self.metric_lst.index(max(self.metric_lst)) if self.mode == 'maximize'  else self.metric_lst.index(min(self.metric_lst))
        if best_epoch != self.best_epoch:
            self.is_best_change = True
            self.best_epoch = best_epoch#update the wether best change flag
        else:
            self.is_best_change = False
        return self.is_best_change

    def best_metric(self):
        if len(self.metric_lst) == 0:return -1
        else:
            return self.metric_lst[self.best_epoch]


def dict_merge(dict_list = []):
    dict_ =  {}
    for dic in dict_list:
        assert isinstance(dic, dict), "object is not a dict!"
        dict_ = {**dict_, **dic}
    return dict_

def get_dataset_path(params = {}):
    dataset = params['dataset']
    file = ws + f'/data/dataset/{dataset}'
    train_path = file + f'/train.npy'
    val_path = file + f'/val.npy'
    test_path = file + f'/test.npy'
    return train_path, val_path, test_path

def write_list_list(fp, list_, model="a", sep=","):
    dir = os.path.dirname(fp)
    if  not os.path.exists(dir): os.makedirs(dir)
    f = open(fp,mode=model,encoding="utf-8")
    count=0
    lines=[]
    for line in list_:
        a_line=""
        for l in line:
            l=str(l)
            a_line=a_line+l+sep
        a_line = a_line.rstrip(sep)
        lines.append(a_line+"\n")
        count=count+1
        if count==10000:
            f.writelines(lines)
            count=0
            lines=[]
    f.writelines(lines)
    f.close()


def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t
    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()

    with open(file_name, "a", newline='\n') as file:  #  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)

#----- Training Utils----------
import argparse
import random, torch
from torch.optim import Adam
from pprint import pprint
from torch.utils.data import DataLoader

def get_common_params():
    # Training settings
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')

    # dataset
    parser.add_argument('--min_task_num', type=int, default=0, help = 'minimal number of task')
    parser.add_argument('--max_task_num',  type=int, default=25, help = 'maxmal number of task')
    parser.add_argument('--dataset', default='logistics_0831', type=str, help='food_cou or logistics')#logistics_0831, logistics_decode_mask
    parser.add_argument('--pad_value', type=int, default=24, help='logistics: max_num - 1, pd: max_num + 1')

    ## common settings for deep models
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 256)')
    parser.add_argument('--num_epoch', type=int, default=60, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 6)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=11, help='early stop at')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 4)')
    parser.add_argument('--task', type=str, default='logistics', help='food_cou or logistics')
    parser.add_argument('--is_eval', type=str, default=False, help='True means load existing model')
    parser.add_argument('--model_path', type=str, default=None, help='best model path in logistics')
    parser.add_argument('--sort_x_size', type=int, default=6)


    return parser

def to_device(batch, device):
    batch = [x.to(device) for x in batch]
    return batch

import nni, time
def train_val_test(train_loader, val_loader, test_loader, model, device, process_batch, test_model, params, save2file):
    model_path = params.get('model_path', None)
    if model_path != None:
        try:
            print('loaded model path:', model_path)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print('best model loaded !!!')
        except:
            print('load best model failed')
        test_result = test_model(model, test_loader, device, params['pad_value'], params, save2file, 'test')
        print('\n-------------------------------------------------------------')
        print(f'{params["model"]} Evaluation in test:', test_result.to_str())
        nni.report_final_result(test_result.to_dict()['krc'])
        return params

    model.to(device)
    optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=params['wd'])
    if params['task'] == 'route':
        early_stop = EarlyStop(mode='maximize', patience=params['early_stop'])
    else:
        early_stop = EarlyStop(mode='minimize', patience=params['early_stop'])

    model_name = model.model_file_name() + f'{time.time()}'
    model_path = ws + f'/data/dataset/{params["model"]}/{params["dataset"]}/sort_model/{model_name}'
    dir_check(model_path)
    for epoch in range(params['num_epoch']):
        if early_stop.stop_flag: break
        postfix = {"epoch": epoch, "loss": 0.0, "current_loss": 0.0}
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as t:
            ave_loss = None
            model.train()
            for i, batch in enumerate(t):
                loss = process_batch(batch, model, device, params)

                if ave_loss is None:
                    ave_loss = loss.item()
                else:
                    ave_loss = ave_loss * i / (i + 1) + loss.item() / (i + 1)
                postfix["loss"] = ave_loss
                postfix["current_loss"] = loss.item()
                t.set_postfix(**postfix)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        if params['is_test']: break

        val_result = test_model(model, val_loader, device, params['pad_value'], params, save2file, 'val')# 对于验证集上，不需要写结果；
        if params['task'] == 'route':
            print('\nval result:', val_result.to_str(), 'Best krc:', round(early_stop.best_metric(),3), '| Best epoch:', early_stop.best_epoch)
            is_best_change = early_stop.append(val_result.to_dict()['krc'])
            if is_best_change:
                # assert val_result.to_dict()['krc'] < early_stop.best_metric(), 'wrong'
                print('value:', val_result.to_dict()['krc'], early_stop.best_metric())
                torch.save(model.state_dict(), model_path)
                print('best model saved')
                print('model path:', model_path)
        else:
            print('\nval result:', val_result.eta_to_str(), 'Best metric:', round(early_stop.best_metric(), 3), '| Best epoch:', early_stop.best_epoch)
            is_best_change = early_stop.append(val_result.eta_to_dict()['mae'])
            if is_best_change:
                print('value:', val_result.eta_to_dict()['mae'], early_stop.best_metric())
                torch.save(model.state_dict(), model_path)
                print('best model saved')
                print('model path:', model_path)

    try:
        print('loaded model path:', model_path)
        model.load_state_dict(torch.load(model_path))
        print('best model loaded !!!')
    except:
        print('load best model failed')
    test_result = test_model(model, test_loader, device, params['pad_value'],params, save2file, 'test')
    print('\n-------------------------------------------------------------')
    print('Best epoch: ', early_stop.best_epoch)
    print(f'{params["model"]} Evaluation in test:', test_result.eta_to_str())

    nni.report_final_result(test_result.to_dict())
    return params

def get_model_function(model):
    model_dict = {}
    import algorithm.ranketpa.route_predictor as ranketpa_route
    model_dict['ranketpa_route'] = (ranketpa_route.PointNet, ranketpa_route.save2file)
    model_dict['ranketpa_time'] = (ranketpa_route.PointNet, ranketpa_route.save2file)
    import algorithm.mlp.mlp as mlp
    model_dict['mlp'] = (mlp.MLP_ETA, mlp.save2file)
    from algorithm.m2g4rtp_delivery.m2g4rtp import M2G4RTP, save2file
    model_dict['m2g4rtp_delivery'] = (M2G4RTP, save2file)
    
    model, save2file = model_dict[model]
    return model, save2file

def run(params, DATASET, PROCESS_BATCH, TEST_MODEL, collate_fn = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device
    params['pad_value'] = params['max_task_num'] - 1

    params['train_path'], params['val_path'],  params['test_path'] = get_dataset_path(params)
    pprint(params)  # print the parameters

    train_dataset = DATASET(mode='train', params=params)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)  # num_workers=2,

    val_dataset = DATASET(mode='val', params=params)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)  # cfg.batch_size

    test_dataset = DATASET(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn)#, collate_fn=collate_fn

    model_save2file = params.get('model_save2file', None) # one can directly pass the model and save2file function to the parameter, without register in the utils
    if  model_save2file is not None:
        model, save2file = model_save2file
    else:
        model, save2file = get_model_function(params['model'])
    model = model(params)
    result_dict = train_val_test(train_loader, val_loader, test_loader, model, device, PROCESS_BATCH, TEST_MODEL, params, save2file)


    return params


if __name__ == '__main__':
    pass
