import pandas as pd
import numpy as np
import geohash2
from geopy.distance import geodesic
from tqdm import tqdm
from utils.util import ws, dir_check, write_list_list, dict_merge, multi_thread_work

def idx(df, col_name):
    _idx_ = list(df.columns).index(col_name)
    return _idx_

def reindex(dic):
    idx = 0
    map_dic = {}
    for k in dic.keys():
        map_dic[k] = idx
        idx += 1
    return map_dic

def time2min(t):
    """
    time string -> date, minute of a day
    :example input: 2021-03-29 17:45:49
    :example output: 29, 1065.8
    """
    M, d = t.split(' ')[0].split('-')
    h, m, s = t.split(' ')[1].split(':')
    return int(f'{M}{d}'), 60 * int(h) + int(m) + int(s) / 60

def split_trajectory(df):
    """
    split the dataframe into trajectories of different couriers
    """
    courier_l = []
    temp = df.values[0]
    c_idx=idx(df, 'courier_id')
    ds_idx =  idx(df, 'ds')
    f = 0
    t = 0
    for row in df.values: #df.values: data in the format of numpy arrary
        if row[c_idx] != temp[c_idx] or row[ds_idx] != temp[ds_idx]:
            courier_l.append(df[f:t]) # pick out the data when enters a new courier id or a new date
            f = t
        t = t + 1
        temp = row
    courier_l.append(df[f:t])
    return courier_l

def check_adjacent_speed(speed_index):
    keep_index = []
    for i in range(len(speed_index) - 1):
        if (speed_index[i] == 1) and (speed_index[i+1] == 1):
            keep_index.extend([False])
        else:
            keep_index.extend([True])
    keep_index.extend([True])
    return keep_index

def drop_unnormal(df, fout):
    # delete abnormal gps drift points
    keep = []
    out_cnt = 0

    courier_l = split_trajectory(df)
    pbar = tqdm(total=len(courier_l))
    for c in courier_l:
        pbar.update(1)
        c = c.reset_index()
        c['speed_adjacent'] = c.apply(lambda r: r['dis_to_last_package'] / r['time_to_last_package'] if r['time_to_last_package'] !=0 else 0, axis=1)
        keep.extend(check_adjacent_speed(((c['speed_adjacent'] > 500).values + 0).tolist()))
    df = df[keep]

    # get couriers features
    couriers, couriers_feature = courier_info(df)

    # filter couriers
    rmv_wd = set(filter(lambda c: couriers_feature['work_days'][c] < 5, couriers))        # work days < 5
    rmv_dmd = set(filter(lambda c: couriers_feature['dis_avg_day'][c] < 50, couriers))   # average travel distance per day < 50m
    rmv_omd = set(filter(lambda c: couriers_feature['order_avg_day'][c] < 5, couriers))  # average package number < 3
    rmv_tmo = set(filter(lambda c: couriers_feature['time_avg_order'][c] < 5, couriers)) # average got time difference between two consecutive packages in the trajectory< 5 min
    rmv_dmo = set(filter(lambda c: couriers_feature['dis_avg_order'][c] < 20, couriers)) # average distance between two two consecutive packages in the trajectory < 20m
    remove_c = rmv_wd & rmv_dmd & rmv_omd & rmv_tmo & rmv_dmo

    # filter courier data
    keep = []
    similar_cnt, remove_cnt = 0, 0
    courier_l = split_trajectory(df)
    pbar = tqdm(total=len(courier_l))
    for c in courier_l:
        pbar.update(1)
        c_v =  c.reset_index()
        for n, row in c_v.iterrows():
            if row['courier_id'] in remove_c:
                keep.append(False)
                remove_cnt += 1
                continue
            if n != 0 and row['dis_to_last_package'] == 0 and row['time_to_last_package'] < 1:  ##consider as one order within one minute
                keep.append(False)
                similar_cnt += 1
                continue
            keep.append(True)

    str_ = f"Remove abnormal GPS  {out_cnt}, filter couriers: couriers/total: {len(remove_c)}/{len(couriers)}, filter number of orders {remove_cnt}, delete redundant order:  {similar_cnt}."
    print(str_)
    write_list_list(fout +'/data_info.txt', [[str_]], 'w')

    report_df = pd.DataFrame({'filter condition': ['work days < 5', 'average travel distance per day < 50m', 'average package number < 3',
                                                   'average got time difference between two consecutive packages in the trajectory< 5 min',
                                                   'average distance between two two consecutive packages in the trajectory < 20m'],
                              'number':[ len(x & remove_c) for x in [rmv_wd, rmv_dmd, rmv_omd, rmv_tmo, rmv_dmo]],
                              'ratio': [  len(x & remove_c) / len(remove_c) if len(remove_c) != 0 else 0 for x in [rmv_wd, rmv_dmd, rmv_omd, rmv_tmo, rmv_dmo]]
                              })
    print('Report of filtered couriers：\n', report_df )

    return df[keep], (couriers, couriers_feature)

def list2str(l):
    # list to str
    return '.'.join(map(str, l))

def str2list(s):
    # str to list
    return [] if s == '' else list(map(int, s.split('.')))

def get_todo_kernel(args: dict):
    result = {}
    def get_a_todo(x):
        now_id = x['index']
        now_time = x['finish_time_minute']
        accepted = c_v[c_v['accept_time_minute'] < now_time]['index']
        df_todo = accepted[accepted > now_id]

        o_id = x['index']
        result[(o_id, 'todo_task')] = list2str(df_todo)
        result[(o_id, 'todo_task_num')] = df_todo.shape[0]
        return x

    c_lst = args['c_lst']
    pbar = tqdm(total=len(c_lst))
    for c in c_lst:
        pbar.update(1)
        c_v = c.reset_index()
        c_v = c_v.apply(lambda x: get_a_todo(x), axis=1)
    return result


def courier_info(df):
    """
    get courier's feature
    """
    couriers=list(set(df['courier_id']))
    # init the feature dict
    feature_dict = {} # courier's feature dict
    for key in ['index', 'id', 'order_sum', 'dis_sum', 'work_days', 'order_avg_day', 'dis_avg_day', 'time_avg_order', 'dis_avg_order', 'speed_avg_order']:
        feature_dict[key] = {}
    # index: index in the courier list, may be used for embedding table
    # order_sum: total order number of a courier in the dataset
    # dis_sum: total travel distance of a courier
    # work_days: work days of a courier
    # order_avg_day: average order number of a courier each day
    # dis_avg_day: average distance per day of a courier
    # time_avg_order: average delta t (in minute) between two consecutive packages of a courier
    # dis_avg_order: average distance (in minute) between two consecutive packages of a courier
    # speed_avg_order: average speed of a courier
    for idx, c in enumerate(couriers):
        c_df = df[df['courier_id'] == c]
        feature_dict['index'][c] = idx
        feature_dict['id'][c] =  c
        feature_dict['order_sum'][c] = c_df.shape[0]
        feature_dict['dis_sum'][c] = sum(c_df['dis_to_last_package'])
        feature_dict['work_days'][c]=len(set(c_df['ds']))
        feature_dict['order_avg_day'][c]=feature_dict['order_sum'][c]/feature_dict['work_days'][c]
        feature_dict['dis_avg_day'][c]=feature_dict['dis_sum'][c]/feature_dict['work_days'][c]
        feature_dict['time_avg_order'][c]=np.mean(c_df['time_to_last_package'])
        feature_dict['dis_avg_order'][c] = np.mean(c_df['dis_to_last_package'])
        feature_dict['speed_avg_order'][c] = feature_dict['dis_sum'][c] / (sum(c_df['time_to_last_package'])) if sum(c_df['time_to_last_package']) != 0 else 5
    return couriers, feature_dict

def process_traj_kernel(args ={}):

    c_lst = args['c_lst']
    pbar = tqdm(total=len(c_lst))

    result = {}
    for c in c_lst:
        pbar.update(1)

        c_v = c.reset_index()
        for n, row in c_v.iterrows():
            date_gt, gt = time2min(row['delivery_time'])  # got_time in minute
            date_at, at = time2min(row['accept_time'])  # accept_time in minute
            if date_gt != date_at:  # got_date != accept_date, means that the order is placed a day another day
                at = at - 60 * 24

            # information of last package
            last_idx = max(0, n - 1)  # last_idx = 0 if n==0 else n-1
            last_ft = c_v.iloc[last_idx]['finish_time_minute']
            last_lon_ = c_v.iloc[last_idx]['lng']
            last_lat_ = c_v.iloc[last_idx]['lat']

            o_id =  row['order_id']
            result[(o_id, 'accept_time_minute')] = at
            result[(o_id, 'time_to_last_package')] = row['finish_time_minute']- last_ft
            result[(o_id, 'dis_to_last_package')] = int(geodesic((last_lat_, last_lon_), (row['lat'], row['lng'])).meters)
    return result

def make_aoi_dict(fin_temp):
    print('start make aoi dict')
    df = pd.read_csv(fin_temp + "/package_feature.csv", sep=',', encoding='utf-8')
    aois = df['aoi_id'].value_counts().to_dict()
    aoi_dict = reindex(aois)
    df['aoi_id'] = df['aoi_id'].apply(lambda x: aoi_dict[x])

    courier_l = split_trajectory(df)

    aoi_nums = df["aoi_id"].nunique()
    aoi_frequency_adj = np.zeros([aoi_nums, aoi_nums])
    aoi_time_adj = np.zeros([aoi_nums, aoi_nums])
    aoi_order_num = np.zeros([aoi_nums])
    aoi_type = np.zeros([aoi_nums])
    aoi_actime = np.zeros([aoi_nums])
    aoi_lng = np.zeros([aoi_nums])
    aoi_lat = np.zeros([aoi_nums])
    pbar = tqdm(total=len(courier_l))

    for i in range(len(courier_l)):
        pbar.update(1)
        for j in range(len(courier_l[i])):
            aoi = courier_l[i].iloc[j]["aoi_id"]
            aoi_order_num[aoi] += 1
            aoi_type[aoi] = courier_l[i].iloc[j]["aoi_type"]
            aoi_actime[aoi] += courier_l[i].iloc[j]["time_to_last_package"]
            aoi_lng[aoi] += courier_l[i].iloc[j]["lng"]
            aoi_lat[aoi] += courier_l[i].iloc[j]["lat"]
            if j != 0:
                from_aoi = courier_l[i].iloc[j - 1]["aoi_id"]
                to_aoi = courier_l[i].iloc[j]["aoi_id"]
                aoi_frequency_adj[from_aoi][to_aoi] += 1
                aoi_time_adj[from_aoi][to_aoi] = aoi_time_adj[from_aoi][to_aoi] + courier_l[i].iloc[j][ "time_to_last_package"]
    aoi_frequency_adj[aoi_frequency_adj == 0] = 1
    aoi_time_adj = np.divide(aoi_time_adj, aoi_frequency_adj)
    aoi_actime = np.divide(aoi_actime, aoi_order_num)
    aoi_lng = np.divide(aoi_lng, aoi_order_num)
    aoi_lat = np.divide(aoi_lat, aoi_order_num)
    aoi_feature = np.zeros([aoi_nums, 5])
    for i in tqdm(range(aoi_nums)):
        aoi_feature[i][0] = i
        aoi_feature[i][1] = aoi_type[i]
        aoi_feature[i][2] = aoi_lng[i]
        aoi_feature[i][3] = aoi_lat[i]
        aoi_feature[i][4] = aoi_actime[i]

    data = {
        "aoi_dict": aoi_dict,
        "aoi_feature": aoi_feature,
        "aoi_time_adj": aoi_time_adj,
        "aoi_frequency_adj": aoi_frequency_adj
    }
    fout = fin_temp + "/aoi_feature.npy"
    print('aoi dict made')
    np.save(fout, data)

def pre_process(fin, fout, is_test=False, thread_num = 20):
    print('Raw input file:' + fin)
    print('Temporary file:' + fout)
    df = pd.read_csv(fin)
    print('length of the original data: ', len(df))

    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    # rank data by date, courier, and got time
    print('Expand basic information...')
    df = df.sort_values(by=['ds', 'courier_id', 'delivery_time'])
    df['finish_time_minute'] = df['delivery_time'].apply(lambda t: time2min(t)[1])

    if is_test:
        df = df[:6000]

    courier_l = split_trajectory(df)

    n = len(courier_l)
    task_num = n // thread_num
    args_lst = [{'c_lst': courier_l[i: min(i+task_num, n)]}   for i in range(0, n, task_num)]
    results = multi_thread_work(args_lst, process_traj_kernel, thread_num)
    result_dict = dict_merge(results)

    init_value = [0] * len(df)
    expand_column = ['accept_time_minute', 'time_to_last_package', 'dis_to_last_package']
    for col in expand_column:
        df[col] = init_value
        df[col] = df['order_id'].apply(lambda x: result_dict[(x, col)])
    print('Basic information expanded...')

    print('Filter data...')
    ## Remove too short trajectories, duplicate packages/orders, and remove GPS drift points, Filter couriers data
    df, courier_information = drop_unnormal(df, fout)
    couriers, couriers_feature  = courier_information
    print('Filter finished...')

    # insert order id

    df.insert(0, 'index', range(1, df.shape[0] + 1))

    print('Get unfinished tasks ...')
    courier_l = split_trajectory(df)
    n =  len(courier_l)
    args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
    results = multi_thread_work(args_lst, get_todo_kernel, thread_num)
    result_dict = dict_merge(results)

    init_value = [0] * len(df)
    expand_column = ['todo_task', 'todo_task_num']
    for col in expand_column:
        df[col] = init_value
        df[col] = df['index'].apply(lambda x: result_dict[(x, col)])
    print('Get unfinished tasks done...')

    # courier's information
    df['dis_avg_day'] = [couriers_feature['dis_avg_day'][c] for c in df['courier_id']]
    df['time_avg_order'] = [couriers_feature['time_avg_order'][c] for c in df['courier_id']]

    print('Update information...')
    courier_l = split_trajectory(df)
    n = len(courier_l)
    task_num = n // thread_num
    args_lst = [{'c_lst': courier_l[i: min(i + task_num, n)]} for i in range(0, n, task_num)]
    results = multi_thread_work(args_lst, process_traj_kernel, thread_num)
    result_dict = dict_merge(results)

    init_value = [0] * len(df)
    expand_column = ['time_to_last_package', 'dis_to_last_package']
    for col in expand_column:
        df[col] = init_value
        df[col] = df['order_id'].apply(lambda x: result_dict[(x, col)])

    df['relative_dis_to_last_package'] = df.apply(lambda r: r['dis_to_last_package'] / r['dis_avg_day'] * 100 if r['dis_avg_day'] !=0 else 0, axis=1)
    df['geohash'] = [geohash2.encode(lat, lon, 8) for lat, lon in zip(df['lat'], df['lng'])]

    print('Features between adjacent tasks constructed...')

    days = sorted(list(set(df['ds'])))
    df['days'] =  [days.index(d) for d in df['ds']]

    ##Generate courier's feature
    cou_df_dic = {}
    for fea, fea_dict in couriers_feature.items():
        fea_lst = [fea_dict[c] for c in couriers]
        cou_df_dic[fea] = fea_lst
    cou_df = pd.DataFrame(cou_df_dic)

    if fout != '':
        dir_check(fout)
        df.to_csv(fout+'package_feature.csv', index=False)
        cou_df.to_csv(fout+'courier_feature.csv', index=False)
        make_aoi_dict(fout)
    print('Data preprocessing is done...')
    return df, cou_df
