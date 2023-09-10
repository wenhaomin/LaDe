# -*- coding: utf-8 -*-
import sys
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from sklearn.multioutput import MultiOutputRegressor
import joblib
from utils.util import *
from utils.eval import Metric
from algorithm.knn.Dataset import KNNDataset

def test(eta_model,  test_dataset, params):
    test_eta_label = []
    test_label_len = []
    test_eta_pred = []

    N = params['max_task_num']
    print('Begin to evaluate')

    iter_num = 5 if params['is_test'] else test_dataset.max_iter
    pbar = tqdm(total=iter_num)

    index = 0
    while index < iter_num:
        pbar.update(1)
        V, V_reach_mask, start_fea, route_label, label_len, time_label = test_dataset[index]

        T = V_reach_mask.shape[0]

        for t in range(T):
            route_label_t = route_label[t, :]
            eta_label = time_label[t, :]

            label_len_t = int(label_len[t])
            if label_len_t == 0:
                continue
            mask = (V_reach_mask[t] == False).astype(int).reshape(-1, 1)
            node_feature = V[t]

            node_fea = (node_feature * mask).reshape(-1)

            start_node_fea = start_fea[t]
            input = np.concatenate([start_node_fea, node_fea])

            pred_eta = eta_model.predict([input])
            eta_pred_t = np.zeros([N])
            for i in range(label_len_t):
                eta_pred_t[i] = pred_eta[0][route_label_t[i]]#用真实路线sort

            test_label_len.append(label_len_t)
            test_eta_label.append(eta_label)
            test_eta_pred.append(eta_pred_t)

        index += 1

    labels_len, eta_label, eta_pred = np.array(test_label_len), np.concatenate(test_eta_label).reshape(-1, N), np.concatenate(test_eta_pred).reshape(-1, N)

    evaluators = [Metric([1, 5]), Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]
    for e in evaluators:
        e.update_eta(labels_len, eta_pred, eta_label)
        print(e.eta_to_str())
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)

    return evaluators[-1]

def main(params):
    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    train_dataset = KNNDataset(mode='train', params=params)
    test_dataset = KNNDataset(mode='test', params=params)

    # construct training data
    input_lst = []
    label_lst = []
    iter_num = 5 if params['is_test'] else test_dataset.max_iter
    index = 0
    while index < iter_num:
        V, V_reach_mask, start_fea, route_label, label_len, time_label = train_dataset[index]

        T = V_reach_mask.shape[0] # init_mask: (T, N)
        for t in range(T):
            eta_label = time_label[t, :]
            route_label_t = route_label[t,:]
            label_len_t = int(label_len[t])
            if label_len_t == 0:
                continue

            #construct, target, feature, label for each step
            mask = (V_reach_mask[t] == False).astype(int).reshape(-1, 1)
            node_fea = V[t]

            node_fea = (node_fea * mask).reshape(-1)
            start_node_fea = start_fea[t]
            input = np.concatenate([start_node_fea, node_fea])

            label = np.zeros(params['max_task_num'])
            for i in range(label_len_t):
                label[route_label_t[i]] = eta_label[i] # feature和label相对应

            input_lst.append(input)
            label_lst.append(label)

        index +=1
    print('Begin to train model...')

    eta_model = MultiOutputRegressor(KNeighborsRegressor(n_neighbors=3))
    eta_model.fit(np.array(input_lst), np.array(label_lst))

    joblib.dump(eta_model, f'{params["dataset"]}_{params["model"]}.pkl')
    eta_model = joblib.load( f'{params["dataset"]}_{params["model"]}.pkl')
    print('model well trained...')

    result_dict = test(eta_model, test_dataset, params)
    print('\n-------------------------------------------------------------')
    print(f'{params["model"]} Evaluation in test:', result_dict.eta_to_str())

    # save the result
    params = dict_merge([result_dict.eta_to_dict(), params])
    return params

def save2file(params):
    from utils.util import save2file_meta, ws
    file_name = ws + f'/output/time_prediction/{params["dataset"]}/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'task', 'eval_min', 'eval_max',
        # model parameters
        'model',
        # training set
        'num_epoch', 'batch_size', 'seed', 'is_test', 'log_time',
        # metric result
        'acc_eta@10', 'acc_eta@20', 'acc_eta@30', 'acc_eta@40', 'acc_eta@50', 'acc_eta@60','mae', 'rmse',
    ]
    save2file_meta(params, file_name, head)

def get_params():
    # Training parameters
    from utils.util import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args


