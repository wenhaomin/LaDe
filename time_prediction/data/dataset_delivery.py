import sys, os, platform
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
from utils.util import ws, dir_check
from data.preprocess_delivery import pre_process, split_trajectory, list2str
import argparse
import copy
from collections import defaultdict
def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)

def idx(df, col_name):
    _idx_ = list(df.columns).index(col_name)
    return _idx_


# we have the pick-up and delivery data
# task, order, package
# worker, courier
# done, todo_, finished, unfinished
# done: means finished task at sample time
# finish_node, unfinish_node, start_node
# done_node, todo_node, start_node

# rem_sorted_list              -> sorted_list
# ref_shuffled_list            -> shuffled_list
# rem_list
#sample_valid_order
# rem_temp
# real_rem_id
# id_first
# three index: index in the dataframe,
# index in the courier's trajectory
# index in the input

np.random.seed(1)

class DeliveryDataset(object):
    def __init__(self, params):
        self.params = params
        self.T = params['T']
        self.N = params['len_range'][1]
        self.N_min = params['len_range'][0]
        self.mode = params['mode']
        self.data =  defaultdict(list)
        self.dic_dis_cal = {}

    def dis_cal(self, lat1, lng1, lat2, lng2):
        # key = str(fro) + '-' + str(to)
        key1 = (lat1, lng1, lat2, lng2)
        key2 = (lat2, lng2, lat1, lng1)
        if key1 not in self.dic_dis_cal.keys():
            distance =  int(geodesic((lat1, lng1), (lat2, lng2)).meters)
            self.dic_dis_cal[key1] = distance

        if key2 not in self.dic_dis_cal.keys():
            distance = int(geodesic((lat1, lng1), (lat2, lng2)).meters)
            self.dic_dis_cal[key2] = distance

        return self.dic_dis_cal[key1]

    def get_features(self, cou, mode):

        _len_ = len(cou)
        if _len_ < self.N_min: return 0  # filter trajectory whose packages less than N_min
        c_v = cou.values
        c_v_samples = cou[1:_len_].values
        c_v = cou.values
        np.random.shuffle(c_v_samples)
        cou_v_shuffle = np.vstack((c_v[0].reshape(1, -1), c_v_samples))
        shuffled_list = cou_v_shuffle[:, self.idx_order_id].tolist() # list of shuffled nodes, index means the shuffled index, value means the true order

        id_first = c_v[0][self.idx_order_id]  # the first order in the trajectory

        V = np.zeros([self.T, self.N, self.params['todo_node_fea_dim']])  # node features
        V_len = np.zeros([self.T])  # number of visible nodes
        V_ft = np.zeros([self.T, self.N])  # finish time of nodes
        V_at = np.zeros([self.T, self.N])  # arrival time of nodes
        V_reach_mask = np.full([self.T, self.N], True)  # mask for reachability of nodes
        V_dispatch_mask = np.zeros([self.T, self.N])  # mask for dispathed nodes
        E_mask = np.zeros([self.T, self.N, self.N])  # mask for edge connection
        E_abs_dis = np.zeros([self.N, self.N])  # edge feature, absolute distance
        E_rel_dis = np.zeros([self.N, self.N])  # edge feature, relative distance
        E_dt_dif = np.zeros([self.N, self.N])  # edge feature, difference of promised pick-up time
        A = np.zeros([self.T, self.N, self.N])  # k-nearest st

        route_label = np.full([self.T, self.N], self.N - 1)  # label of route prediction for  each step
        route_label_all = np.full([self.T, self.N], self.N - 1)
        route_label_len = np.zeros([self.T])  # number of valid label nodes for each step
        eta_label_len =  np.zeros([self.T])
        eta_label_td = np.zeros([self.T, self.N])  # label: time difference

        start_fea = np.zeros([self.T, self.params['start_node_fea_dim']])  # features at start node: (dt, lng, lat, pt_last, ft)
        start_idx = np.zeros([self.T])  # index of start node
        past_x = np.zeros([self.T, self.params['done_node_num'], self.params['done_node_fea_dim']])  # The features of the previous [done_node_num]-packages of each step
        cou_fea = list(np.array(self.df_cou.loc[self.df_cou['id'] == c_v[0][self.idx_courier_id]]).reshape(-1)[[0, 2, 3, 4, 5, 6, 7, 8, 9]])  # (id, work_days)

        sample_valid_task = []
        t = 0  # the current step
        for start in range(len(c_v)):  # time steps are decided by finish time
            if t == self.T:
                continue

            unpick_set = str(c_v[start][self.idx_todo_task])
            if (c_v[start][self.idx_todo_task_num] <= 0) or (unpick_set == 'nan'): continue  # move to the next episode of the trajectory

            todo_lst = unpick_set.split('.')
            remove_lst = []

            # filter packages
            for task in todo_lst:
                task_id = int(task)
                c_idx = task_id - id_first  # (pre_end + 1) tasks
                if c_idx <= start:
                    remove_lst.append(task)
                    print('Error of start index')
                    continue
                # max number of tasks considered
                if c_idx >= self.N - 1:
                    remove_lst.append(task)
                    if mode == 'test':
                        # print('remove sample that are out of index bound')
                        return 0
                    continue
                # filter packages according to time label
                time_label = c_v[c_idx][self.idx_finish_time] - c_v[start][self.idx_finish_time]
                if time_label <= self.params['label_range'][0] or time_label > self.params['label_range'][1]:
                    remove_lst.append(task)
                    continue

            for x in remove_lst:
                todo_lst.remove(x)

            if len(todo_lst) <= self.N_min:  # remove sample with short length
                continue
            #############################   #############################   #############################
            # After the filtering operation, start to extract features
            courier_id = []  # Integer
            query_time = []  # string
            unfinished_task = []  # string

            # The smaller the index, the earlier it is picked, and the sorted list is obtained by sorting according to the actual picked time.
            todo_rank = list(map(int, todo_lst))
            todo_rank.sort()

            step_valid_order = list(map(int, todo_lst))  # Build features and masks based on valid orders for each step
            sample_valid_task.extend(step_valid_order)
            for task in todo_rank:
                idx = shuffled_list.index(task)  # return the index of the task in the shuffled list
                V_reach_mask[t, idx] = False
                V_dispatch_mask[[x for x in range(t, self.T)], idx] = 1  # remove the mask of accepted tasks
                unfinished_task.append(cou[cou['index'] == task]['order_id'].iloc[0])

            V_dispatch_mask[[x for x in range(t, self.T)], shuffled_list.index(c_v[:, self.idx_order_id][start])] = 1  # The start node is considered as accepted

            current_time = c_v[start][self.idx_finish_time]  # the time of the current step
            # find the most closed accept_time to current_time
            close_time_dif = np.inf
            close_accept_time = np.inf
            for i in range(len(c_v)):
                accept_t = c_v[i][self.idx_accept_time]
                if (accept_t - current_time > 0) and (accept_t - current_time < close_time_dif):
                    close_time_dif = accept_t - current_time
                    close_accept_time = accept_t

            # For label construction, remove package which is influenced by new comming packages, i.e., got_time < close_accept_time
            todo_rank_ = copy.deepcopy(todo_rank)
            if close_accept_time != np.inf:
                todo_rank = list(filter(lambda k: c_v[k - id_first][self.idx_finish_time] <= close_accept_time, todo_rank))

            for task in todo_rank: #filtered tasks
                route_label[t, todo_rank.index(task)] = shuffled_list.index(task)
            for task in todo_rank_:
                task_idx = task - id_first
                eta_label_td[t, todo_rank_.index(task)] = c_v[task_idx][self.idx_finish_time] - c_v[task_idx - 1][self.idx_finish_time]
                V_at[t, todo_rank_.index(task)] = c_v[task_idx][self.idx_finish_time] - c_v[start][self.idx_finish_time] # actual arrival time of all tasks
                route_label_all[t, todo_rank_.index(task)] = shuffled_list.index(task)

            route_label_len[t] = len(todo_rank)
            eta_label_len[t] = len(todo_rank_)
            # For test set construction
            courier_id.append(cou['courier_id'].iloc[0])

            # iterate all accepted orders, and connect them in the graph
            for i in range(self.N):
                for j in range(self.N):
                    if (V_dispatch_mask[t][i] == 1) and (V_dispatch_mask[t][j] == 1):  # V_dispatch_mask (T, N)
                        E_mask[t, i, j] = 1
                        E_mask[t, j, i] = 1

                # construct features
            for task in todo_lst:
                task_id = int(task)
                c_idx = task_id - id_first
                # The distance between each package and the start point
                dis_abs = self.dis_cal(c_v[start][self.idx_latitude], c_v[start][self.idx_longitude],
                                  c_v[c_idx][self.idx_latitude], c_v[c_idx][self.idx_longitude])
                idx = shuffled_list.index(task_id)

                # contruct edge feature
                E_abs_dis[0, idx] = dis_abs
                E_abs_dis[idx, 0] = dis_abs

                if c_v[c_idx][self.idx_courier_distance] == 0:
                    dis_temp = 0.0
                else:
                    dis_temp = dis_abs / c_v[c_idx][self.idx_courier_distance] * 100

                E_rel_dis[0, idx] = dis_temp
                E_rel_dis[idx, 0] = dis_temp

                # contruct edge feature
                E_dt_dif[0, idx] = c_v[0][self.idx_accept_time] - c_v[c_idx][self.idx_accept_time]
                E_dt_dif[idx, 0] = c_v[c_idx][self.idx_accept_time] - c_v[0][self.idx_accept_time]
                node_feature = c_v[c_idx][[self.idx_accept_time, self.idx_longitude, self.idx_latitude, self.idx_type_code]].tolist()
                node_feature.append(dis_temp)
                node_feature.append(dis_abs)
                V[t, idx, :] = np.array(node_feature)

                V_ft[t, idx] = c_v[c_idx][self.idx_finish_time]
                # get feature of start node, accept_time, lng, lat, pt, ft
            start_fea[t, :] = c_v[start][[self.idx_accept_time, self.idx_longitude, self.idx_latitude,self.idx_finish_time]]
            start_idx[t] = shuffled_list.index(c_v[:, self.idx_order_id][start])
            query_time.append(c_v[start][self.idx_abs_got_time])

            # get features of done_node_num packages
            s, e = max(0, start-2), start + 1 # window start and end
            for idx, n in enumerate(range(s, e)):
                done_feature = [c_v[n][self.idx_finish_time], c_v[n][self.idx_accept_time],
                                 c_v[n][self.idx_longitude], c_v[n][self.idx_latitude], c_v[n][self.idx_type_code],
                                 c_v[n][self.idx_relative_dis_to_last_package], c_v[n][self.idx_dis_to_last_package],
                                 c_v[n][self.idx_time_to_last_package]]
                past_x[t, idx, :] = np.array(done_feature)

            t += 1

            # Traversing all steps finished
            ############  ############  ############  ############  ############  ############  ############  ############
        if len(list(set(sample_valid_task))) == 0:  # no valid packages, return
            return 0

        sample_valid_task = list(set(sample_valid_task))
        V_len[:] = (np.array(sample_valid_task) - id_first).max()
        for i in sample_valid_task:
            for j in sample_valid_task:
                idx_i = shuffled_list.index(i)  #
                idx_j = shuffled_list.index(j)  #
                c_idx_i = i - id_first
                c_idx_j = j - id_first
                E_dt_dif[idx_i][idx_j] = c_v[c_idx_i][self.idx_accept_time] - c_v[c_idx_j][self.idx_accept_time]
                E_dt_dif[idx_j][idx_i] = c_v[c_idx_j][self.idx_accept_time] - c_v[c_idx_i][self.idx_accept_time]
                dis_abs = self.dis_cal(c_v[c_idx_i][self.idx_latitude], c_v[c_idx_i][self.idx_longitude],
                                  c_v[c_idx_j][self.idx_latitude], c_v[c_idx_j][self.idx_longitude])
                if c_v[c_idx_i][self.idx_courier_distance] == 0:
                    dis_temp = 0.0
                else:
                    dis_temp = dis_abs / c_v[c_idx_i][self.idx_courier_distance] * 100
                E_abs_dis[idx_i][idx_j] = dis_abs
                E_abs_dis[idx_j][idx_i] = dis_abs
                E_rel_dis[idx_i][idx_j] = dis_temp
                E_rel_dis[idx_j][idx_i] = dis_temp

        A_reach = ~V_reach_mask + 0
        for t in range(self.T):
            cur_idx = start_idx[t]
            reachable_nodes = np.argwhere(A_reach[t] == 1).reshape(-1)
            reachable_nodes = np.append(reachable_nodes, [cur_idx])
            for i in range(self.N):
                if i in reachable_nodes:
                    for j in range(self.N):
                        if j in reachable_nodes:
                            if i != j:
                                A[t][i][j] = 1
                            else:
                                A[t][i][j] = -1

        for t in range(self.T):
            for i in range(self.N):
                if len(np.argwhere(A[t][i] == 1).reshape(-1)) < 5:
                    continue
                dis_from_i = E_abs_dis[i] * A[t][i]
                remove_dis_idx = np.argsort(abs(dis_from_i))[-1]
                A[t][i][[remove_dis_idx]] = 0

        #concatenate edge feature
        E_fea_lst = [np.expand_dims(x, axis=-1) for x in [E_abs_dis, E_rel_dis, E_dt_dif]]
        E_static_fea = np.concatenate(E_fea_lst, axis=2)

        return {'V': V, 'E_mask': E_mask, 'E_static_fea': E_static_fea, 'V_reach_mask': V_reach_mask, 'V_dispatch_mask': V_dispatch_mask,
                'E_abs_dis': E_abs_dis, 'E_rel_dis': E_rel_dis, 'E_dt_dif': E_dt_dif, 'route_label': route_label,
                'label_len': route_label_len, 'V_len': V_len, 'start_fea': start_fea, 'start_idx': start_idx, 'past_x': past_x,
                'cou_fea': cou_fea, 'V_ft': V_ft, 'time_label': V_at, 't_interval': eta_label_td, 'A': A}

    def results_merge(self, results):

        all_result = []
        for r in results:
            all_result += r

        feature_lst = ['V', 'V_len', 'V_ft', 'V_reach_mask', 'V_dispatch_mask',  # node/task related information
                       'E_mask', 'A', 'E_static_fea',  # edge related information
                       'start_fea', 'start_idx',  # start node information,
                       'past_x', 'cou_fea',  # past task, and courier's feature
                       'route_label', 'time_label', 'label_len', 't_interval']  # label information,

        for r in all_result:
            for f in feature_lst:
                self.data[f].append(r[f])

        for f in feature_lst:
            self.data[f] = np.stack(self.data[f], axis=0)

        return self.data

    def multi_thread_work(self, parameter_queue, function_name, thread_number=5):
        from multiprocessing import Pool
        """
        For parallelization
        """
        pool = Pool(thread_number)
        result = pool.map(function_name, parameter_queue)
        pool.close()
        pool.join()
        return result

    def get_feature_kernel(self, args={}):
        c_lst = args['c_lst']
        pbar = tqdm(total=len(c_lst))
        result_lst = []

        for cou in c_lst:
            pbar.update(1)
            win_num = (2 + (len(cou) - (self.N - 1)) // self.T) if len(cou) >= (self.N - 1) else 1
            for i in range(win_num):
                if i == win_num - 1:
                    result = self.get_features(cou[i * self.T:], self.mode)
                else:
                    result = self.get_features((cou[i * self.T: i * self.T + (self.N - 1)]), self.mode)

                if result == 0:
                    continue
                else:
                    result_lst.append(result)
        return result_lst

    def get_data(self):
        if os.path.exists(self.params['fin_temp']):
            df = pd.read_csv(self.params['fin_temp'] + "/package_feature.csv", sep=',', encoding='utf-8')  # package features
            self.df_cou = pd.read_csv(self.params['fin_temp'] + "/courier_feature.csv", sep=',', encoding='utf-8')  # courier features
        else:
            df, self.df_cou = pre_process(fin=self.params['fin_original'], fout=self.params['fin_temp'], is_test=self.params['is_test'])
        # feature index in the dataframe
        self.idx_order_id = idx(df, 'index')
        self.idx_block_id = idx(df, 'region_id')
        self.idx_courier_id = idx(df, 'courier_id')
        self.idx_todo_task_num = idx(df, 'todo_task_num')
        self.idx_todo_task = idx(df, 'todo_task')
        self.idx_finish_time = idx(df, 'finish_time_minute')
        self.idx_accept_time = idx(df, 'accept_time_minute')
        self.idx_longitude = idx(df, 'lng')
        self.idx_latitude = idx(df, 'lat')
        self.idx_type_code = idx(df, 'aoi_type')
        self.idx_courier_distance = idx(df, 'dis_avg_day')
        self.idx_relative_dis_to_last_package = idx(df, 'relative_dis_to_last_package')
        self.idx_dis_to_last_package = idx(df, 'dis_to_last_package')
        self.idx_time_to_last_package = idx(df, 'time_to_last_package')
        self.idx_abs_got_time = idx(df, 'delivery_time')
        all_dates = df['ds'].unique()
        train_dates = all_dates[: int(len(all_dates) * self.params['train_ratio'])].tolist()
        val_dates = all_dates[int(len(all_dates) * self.params['train_ratio']) + 1: int(len(all_dates) * (self.params['train_ratio'] + self.params['val_ratio']))].tolist()
        test_dates = all_dates[int(len(all_dates) * 0.8) + 1:].tolist()
        if self.params['is_test'] == False:

            if self.mode == 'train':
                df = df[df['ds'].isin(train_dates)]
            elif self.mode == 'val':
                df = df[df['ds'].isin(val_dates)]
            elif self.mode == 'test':
                df = df[df['ds'].isin(test_dates)]
            else:
                raise RuntimeError('please specify a mode')

        courier_l = split_trajectory(df)
        thread_num = self.params['num_thread']
        n = len(courier_l)
        task_num = n // thread_num
        args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
        results_list = self.multi_thread_work(args_lst, self.get_feature_kernel, self.params['num_thread'])
        return self.results_merge(results_list)

def get_params():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--fin_temp', type=str, default=ws + '/data/tmp/delivery_cq/')
    parser.add_argument('--fin_original',  type=str, default=ws + '/data/raw/delivery/delivery_cq.csv')
    parser.add_argument('--data_name', type=str, default='delivery_cq')
    parser.add_argument('--num_thread', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=1)

    parser.add_argument('--is_test', type=str, default=False)
    parser.add_argument('--block', type=dict, default={0}, help='0 means selecting all blocks, or specify other blocks')
    parser.add_argument('--label_range', type=tuple, default=(0, 120))
    parser.add_argument('--len_range', type=tuple, default=(1, 25))
    parser.add_argument('--T', type=int, default=12, help='the max number of valid time step in a sample')
    parser.add_argument('--todo_node_fea_dim', type=int, default=6, help='feature number for each unfinished task node')
    parser.add_argument('--start_node_fea_dim', type=int, default=4, help='feature number for start node')
    parser.add_argument('--done_node_fea_dim', type=int, default=8, help='feature number for each finished node')
    parser.add_argument('--done_node_num', type=int, default=3, help='the number of last finished nodes')

    # data split, according to date
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    args, _ = parser.parse_known_args()
    return args

def main():
    params = vars(get_params())
    if params['is_test']: params['data_name'] += '_test'
    data_name = params['data_name']

    for mode in [ 'test', 'train', 'val']:  # 'train', 'val', 'test'
        if mode == 'train':
            fout = ws + f'/data/dataset/{data_name}/train.npy'
            dir_check(fout)
            params['mode'] = mode
            DataProcess = DeliveryDataset(params)
            np.save(fout, DataProcess.get_data())
            print('train file saved at: ', fout)
        elif mode == 'val':
            fout = ws + f'/data/dataset/{data_name}/val.npy'
            dir_check(fout)
            params['mode'] = mode
            DataProcess = DeliveryDataset(params)
            np.save(fout, DataProcess.get_data())
            print('val file saved at: ', fout)
        else:
            fout = ws + f'/data/dataset/{data_name}/test.npy'
            dir_check(fout)
            params['mode'] = mode
            DataProcess = DeliveryDataset(params)
            np.save(fout, DataProcess.get_data())
            print('test file saved at: ', fout)

    print('Dataset constructed...')

if __name__ == "__main__":
    main()
