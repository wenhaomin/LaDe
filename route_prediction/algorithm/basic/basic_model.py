# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from pprint import pprint
from typing import List, Optional

from utils.eval import Metric
from utils.util import to_device, ws, dict_merge, get_dataset_path, get_nonzeros_nrl

class BaselineDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,
    )->None:
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train':'train_path', 'val':'val_path','test':'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):
        return len(self.data['V_len'])

    def __getitem__(self, index):

        V = self.data['V'][index]
        V_reach_mask = self.data['V_reach_mask'][index]

        E_static_fea = self.data['E_static_fea'][index]
        E_abs_dis = E_static_fea[:, :, 0] # the absolute distance between two tasks

        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]

        label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]

        return V, V_reach_mask, E_abs_dis, start_fea, start_idx, label, label_len


class TimeGreedyModel(nn.Module):
    def __init__(self, params):
        super(TimeGreedyModel, self).__init__()

    # mask=True, when it cannot be outputted.
    def forward(self, time, mask, pad_value):
        batch_size, outputs, max_len = time.shape[0], [], time.shape[1]
        pred_len = []
        for i in range(batch_size):
            t,  msk, pred = time[i], mask[i], np.full(max_len, pad_value)
            t = t.squeeze()
            j = 0
            while not msk.all():
                t_j = t.masked_fill(msk, 1e6)
                idx = torch.argmin(t_j)
                pred[j], msk[idx] = idx, 1
                j += 1
            outputs.append(list(map(int, pred)))
            pred_len.append(j)
        return outputs, pred_len


class DistanceGreedyModel(nn.Module):
    def __init__(self, params):
        super(DistanceGreedyModel, self).__init__()

    def forward(self, distance, mask, start_idx, pad_value):
        """
        :param distance:  (B, N, N), distance matrix
        :param mask:      (B, N),    init mask, mask nodes that are infeasible
        :param start_idx: (B,)       index of start index
        :param pad_value:  int,       padding value in the output
        :return:
        """
        batch_size, outputs, max_len = distance.shape[0], [], distance.shape[1]
        pred_len = []
        for i in range(batch_size):
            dis, msk, pred, point = distance[i], mask[i], np.full(max_len, pad_value), start_idx[i]  # pad value of pred should be the same as target
            j = 0
            while not msk.all():
                dis_j = dis[point].masked_fill(msk, 1e6)
                idx = torch.argmin(dis_j)
                # if idx % 2 != 0:
                #     msk[idx + 1] = 0
                pred[j], msk[idx], point = idx, 1, idx
                j += 1
            outputs.append(list(map(int, pred)))
            pred_len.append(j)

        return outputs, pred_len


from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
class OrtoolsModel(nn.Module):

    def __init__(
            self, params
    ) -> None:
        super().__init__()
        self.inf = nn.Parameter(torch.tensor([np.inf], dtype=torch.float32), requires_grad=False)
        self.model_name = 'Google OR-Tools'

    def dist(self, p1, p2, is_lat_first=True):
        from geopy.distance import geodesic
        if is_lat_first:
            d = geodesic(p1, p2).m
        else:
            d = geodesic((p1[1], p1[0]), (p2[1], p2[0]))
        return d

    def create_distance_callback(self, dist_matrix):
        # Create a callback to calculate distances between cities.

        def distance_callback(from_node, to_node):
            return int(dist_matrix[from_node][to_node])

        return distance_callback

    def ortools_best_route(self, points, cur):
        from geopy.distance import geodesic
        area_names = []
        all_area_lon_lat = []
        dist_matrix = []
        area_names.append(0)
        all_area_lon_lat.append(cur)#start point
        for i in range(0, len(points)):
            area_names.append(i + 1)
            all_area_lon_lat.append(points[i])#start, points
        for i in range(0, len(all_area_lon_lat)):#iterate all points
            temp = []
            for j in range(0, len(all_area_lon_lat)):
                temp.append(
                    self.dist((all_area_lon_lat[i][0], all_area_lon_lat[i][1]),
                              (all_area_lon_lat[j][0], all_area_lon_lat[j][1]),
                              is_lat_first=False) / geodesic(1) * 1000)
            dist_matrix.append(temp)
        tsp_size = len(area_names)  # node number
        num_routes = 1  # route number
        start = 0  # the start and end node of the route
        distance = 100000
        route = []
        for i in range(1, len(area_names)):
            routing = pywrapcp.RoutingModel(tsp_size, num_routes, [start], [i])#tsp_size: 7, num_routes: 1, start: 0, i: 0
            search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
            # calculate the distance between two nodes
            dist_callback = self.create_distance_callback(dist_matrix)
            routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)
            # Solve the problem.
            assignment = routing.SolveWithParameters(search_parameters)
            if assignment:
                if assignment.ObjectiveValue() < distance:
                    distance = assignment.ObjectiveValue()
                    index = routing.Start(start)
                    route_temp = []
                    while not routing.IsEnd(index):
                        # Convert variable indices to node indices in the displayed route.
                        # IndexToNode: index of the current node
                        # NextVar
                        route_temp.append(area_names[routing.IndexToNode(index)] - 1)
                        index = assignment.Value(routing.NextVar(index))
                    route_temp.append(area_names[routing.IndexToNode(index)] - 1)
                    route = route_temp
        if len(route) != 0: route.remove(-1)
        return route

    def forward(
            self,
            src: Tensor,  # [B, N, F] F is the feature length
            start: Tensor,  # [B, F'] F' is the feature length
            mask: Optional[Tensor] = None,  # [B, N]
    ) -> List:
        predict_seq = []
        pred_seq_len = []
        src = src.detach()
        start = start.detach()
        mask = mask.detach()
        length = mask.shape[0]
        predict_idx = np.full([length, mask.size(-1)], 24)
        for i in range(length):#iterate each sample, which outpus a single route
            points2 = []
            pred_len = []
            num_mask = 25 # this should be changed when the maximial task number changes
            N = 25
            value_list = []
            for j in range(N):
                if not mask[i, j]: # if not masked
                    points2.append((src[i, j, [0, 1]]).numpy().tolist())
                    value_list.append(j)
                    num_mask -= 1
            start2 = (start[i, [0, 1]]).numpy().tolist()#start point of each step
            predict = self.ortools_best_route(points2, start2)# [1,2,3,0,4,5]
            pred_len.append(len(predict))
            for v in range(len(predict)):
                predict_idx[i][v] = value_list[predict[v]]

            # for j in range(num_mask):
            #     predict.append(24)
            # predict_seq.append(torch.Tensor(predict))
            pred_seq_len.append(pred_len)

        pred_seq_len = np.stack(pred_seq_len, axis = 0).reshape(-1)

        return predict_idx,  pred_seq_len

def test_model(model, test_loader, device, params):
    evaluators = [Metric([1, 5]), Metric([1, 11]), Metric([1, 15]), Metric([1, 25])]
    params['pad_value'] = params['max_task_num'] - 1

    total_len = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = to_device(batch, device)

            V, V_reach_mask, E_abs_dis, start_fea, start_idx, label, label_len = batch


            B, T, N = V_reach_mask.size()
            B_T = B * T
            seq_dis = torch.repeat_interleave(E_abs_dis.unsqueeze(1), repeats=T, dim=1).reshape(B_T, N, N)  # (B * T, N, N)
            seq_time = V[:, :, :, 7].unsqueeze(3).reshape(B_T, N, 1) # note: 7 is the feature index of time requirement

            length_target = label_len.reshape(-1)
            init_mask = V_reach_mask.reshape(-1, N)
            target = label.reshape(B_T, -1)

            if params['model'] == 'Distance-Greedy':
                # seq_dis: (B_T, N, N), init_mask:(B_T, N), start_idx: (B_T, N)
                start_idx = start_idx.reshape(-1).long()
                output, length_pred = model(seq_dis, init_mask, start_idx,  params['pad_value'])
            elif params['model'] == 'Time-Greedy':
                output, length_pred = model(seq_time, init_mask, params['pad_value'])
            if params['model'] == 'Or-Tools':
                src = V[:, :, :, [1, 2]].reshape(B_T, N, -1)  # (B*T, N, 2) lng, lat
                start = start_fea[:, :, [1, 2, 4]].reshape(B_T, -1)  # (B*T, feat_len'(lng, lat, ft))
                output, length_pred = model(src, start, init_mask)

            output = torch.LongTensor(output).to(target.device)
            length_pred = torch.LongTensor(length_pred).to(target.device)
            pred, label, label_len, pred_len = get_nonzeros_nrl(output, target, length_target, length_pred, params['pad_value'])
            # [print(f'pred:{a}, target:{b}, length nodes:{c}, length target:{d}') for a, b, c, d in zip(pred.numpy().tolist(), label.cpu().numpy().tolist(), pred_len, label_len)]
            add_len = pred_len.clone()
            total_len = total_len + add_len.cpu().tolist()
            for e in evaluators:
                e.update(pred, label, label_len, pred_len)

    print('test step_sample_num:', len(np.array(total_len).reshape(-1)))
    print('sum of step sample num:', (np.array(total_len).reshape(-1)).sum())

    for e in evaluators:
        print(e.to_str())
        params_save = dict_merge([e.eta_to_dict(), params])
        params_save['eval_min'], params_save['eval_max'] = e.len_range
        save2file(params_save)

    return evaluators[-1].to_dict()


from utils.util import save2file_meta

def save2file(params):
    file_name = ws + f'/output/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'task', 'eval_min', 'eval_max',
        # model parameters
        'model',
        # training set
        'num_epoch', 'batch_size', 'seed', 'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',
    ]
    save2file_meta(params, file_name, head)



def get_model_function(model):
    if model == 'Distance-Greedy':
        return DistanceGreedyModel, save2file
    elif model == 'Time-Greedy':
        return TimeGreedyModel, save2file
    elif model == 'Or-Tools':
        return OrtoolsModel, save2file
    else:
        raise NotImplementedError


def main(params):
    params['train_path'], params['val_path'], params['test_path'] = get_dataset_path(params)
    pprint(params)

    test_dataset = BaselineDataset(mode='test', params=params)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    device = torch.device('cpu')

    model, save2file = get_model_function(params['model'])
    model = model(params)
    result_dict = test_model(model, test_loader, device, params)
    params = dict_merge([result_dict, params])
    # save2file(params)
    return params


def get_params():
    from utils.util import get_common_params
    parser = get_common_params()
    # Model parameters
    parser.add_argument('--model', type=str, default='Time-Greedy')  # Or-Tools, Distance-Greedy, Time-Greedy
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    import time, nni
    import logging

    logger = logging.getLogger('training')
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        for model in ['Distance-Greedy', 'Time-Greedy']:
            params['model'] = model
            params['dataset'] = 'pickup_jl'
            params['batch_size'] = 16
            params.update(tuner_params)
            main(params)
    except Exception as exception:
        logger.exception(exception)
        print(exception)
        raise