# -*- coding: utf-8 -*-
import torch

import lightgbm as lgb
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)


from utils.util import *
from utils.eval import Metric
from algorithm.osqure.Dataset import OsqureDataset

def test(classifier, test_dataset, params, log_path = ''):
    result_route = []
    target_route = []
    target_len_l = []
    seq_len_l = []
    max_num_nodes = params['max_task_num']
    pad_value = params['max_task_num'] - 1

    iter_num = 50 if params['is_test'] else test_dataset.max_iter
    index = 0

    while index < iter_num:
    # while index < 30:
        V, V_reach_mask, label, label_len = test_dataset[index]
        x_nodes_features = V  # (T, N, 8) , features of all nodes in a step
        init_mask = V_reach_mask
        target = label

        #  the distance and time feature will be updated
        max_steps = init_mask.shape[0]
        for t in range(max_steps):
            target_step = target[t, :]
            if target_step.min() == pad_value:  # no target in this step
                continue

            init_mask_step = (init_mask[t] == False).astype(int).reshape(-1, 1)
            feature_step_i = x_nodes_features[t]
            valid_target_len = list(target_step).index(pad_value)
            pred_route = []

            # Traversing the valid values of the valid step
            while init_mask_step.any():
                out_step_i = np.zeros([1, classifier._n_classes + 1])
                feature_step_i_masked = feature_step_i * init_mask_step

                out_step_i[0, 1:] = classifier.predict_proba(feature_step_i_masked.reshape(1, -1))
                pred_loc = np.argmax(((out_step_i + 0.01).reshape(-1, 1)) * init_mask_step[:classifier._n_classes + 1])
                #update mask according to prediction
                pred_route.append(pred_loc)
                init_mask_step[pred_loc] = 0

            seq_len_l.append(len(pred_route))
            #pad current route and record the pred and target
            for j in range(len(pred_route), max_num_nodes):
                pred_route.append(pad_value)
            result_route.append(pred_route)
            target_route.append(target_step)
            target_len_l.append(valid_target_len)
        index += 1

    pred_steps, label_steps, labels_len, preds_len = np.concatenate(result_route).reshape(-1,max_num_nodes), \
                                                 np.concatenate(target_route).reshape(-1, max_num_nodes), \
                                                 np.array(target_len_l), np.array(seq_len_l)


    evaluators = [Metric([1, 5]),  Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]
    for e in evaluators:
        e.update(pred_steps, label_steps, labels_len, preds_len)
        print(e.to_str())
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)

    return evaluators[-1]

def main(params):
    params['model'] = 'osqure'
    params['feature_num'] = 8
    params['hidden_size'] = 0

    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    train_dataset = OsqureDataset(mode='train', params=params)
    test_dataset = OsqureDataset(mode='test', params=params)
    max_num_nodes = params['max_task_num']
    pad_value = params['max_task_num'] - 1
    train_feature = []
    train_label = []

    iter_num = 50 if params['is_test'] else train_dataset.max_iter
    index = 0
    while index < iter_num:
        V, V_reach_mask, label, label_len = train_dataset[index]
        x_nodes_features = V  # (T, N, 8) , features of all nodes in a step
        init_mask = V_reach_mask
        target = label

        max_steps = init_mask.shape[0]
        for t in range(max_steps):
            target_step = target[t,:]
            if target_step.min() == pad_value:
                continue
            # target, feature, label
            init_mask_step = (init_mask[t] == False).astype(int).reshape(-1, 1)
            feature_step_i = x_nodes_features[t]
            valid_target_len = list(target_step).index(pad_value)#
            # loc_order_num = step_order_num[t].reshape(-1, 1)

            for i in range(valid_target_len):#output a route
                feature_step_i_masked = feature_step_i * init_mask_step
                train_feature.append(feature_step_i_masked.reshape(-1))#  #all nodes features of a step
                train_label.append(target_step[i])

                #update mask
                init_mask_step[target_step[i]] = 0
        index +=1
    print('train data loaded...')
    train_feature_set = np.concatenate(train_feature).reshape(-1,  max_num_nodes * params['feature_num'])
    train_label_set = np.array(train_label) #predict one location for a time，27 * 5 -> 1 label
    bst = lgb.LGBMClassifier(use_missing = True, zero_as_missing = True)
    bst.fit(train_feature_set, train_label_set)
    print('model well trained...')

    result_dict = test(bst, test_dataset, params)  #
    print('\n-------------------------------------------------------------')
    print(f'{params["model"]} Evaluation in test:', result_dict.to_str())

    # save the result
    params = dict_merge([result_dict.to_dict(), params])
    # save2file(params)

    return params

# ---Log--
from utils.util import save2file_meta
def save2file(params):
    from utils.util import ws
    file_name = ws + f'/output/{params["model"]}.csv'
    # 写表头
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',

    ]
    save2file_meta(params, file_name, head)

def get_params():
    # Training parameters
    from utils.util import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":

    import time, nni
    import logging


    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print('GPU:', torch.cuda.current_device())
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        params['dataset'] = 'pickup_jl'
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
