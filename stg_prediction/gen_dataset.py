import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle as pkl

#################################
city_name = 'Shanghai' ## choose from ['Shanghai', 'Hangzhou', 'Chongqing']
saved_dir = './data/Delivery_SH'
raw_data_path = './data/pickup_sh.csv'
seq_length_x = 24
seq_length_y = 24
train_ratio = 0.6
val_ratio = 0.2
interval = 1
# temporal granularity
n_time = '1h' 
obj_type = 'poi'   ## choose from ['poi', 'got_gps', 'accept_gps']
##################################
# read data
df_raw = pd.read_csv(raw_data_path)

if obj_type == 'poi':
    obj_time = 'pickup_time'
elif obj_type == 'got_gps':
    obj_time = 'pickup_gps_time'
# save dir
if not os.path.exists(saved_dir):os.mkdir(saved_dir)

# chose city
df_city = df_raw[df_raw.city == city_name]
df_city[obj_time] = pd.to_datetime(df_city[obj_time], format='%m-%d %H:%M:%S')
df_city = df_city.sort_values(obj_time)

groups = df_city.groupby(pd.Grouper(key=obj_time, freq=n_time))
sub_dfs = [group[1] for group in groups]
time_feature = [group[0] for group in groups]
time_h = np.array([i.hour for i in time_feature])
time_m = np.array([i.minute for i in time_feature])
print('Number of time stamps:{}'.format(len(time_feature)))

# a list contain dipan's id
dipan_ids = list(df_city.region_id.unique())

dataset = np.zeros(shape=(len(time_h), len(dipan_ids),1))
for _, sub_df in enumerate(sub_dfs):
    for i, dipan in enumerate(dipan_ids):
        df = sub_df[sub_df.region_id == dipan]
        dataset[_, i, 0] = len(df)

print(dataset.shape)

################## Split dataset #########################
x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1) * interval, 1, interval),)))
y_offsets = np.sort(np.arange(interval, (seq_length_y + 1) * interval, interval))

num_samples = dataset.shape[0]

x, y = [], []
timeH_x, timeH_y = [], []
timeM_x, timeM_y = [], []
min_t = abs(min(x_offsets))
max_t = abs(num_samples - abs(max(y_offsets))) 
for i, t in enumerate(range(min_t, max_t)):  # t is the index of the last observation.
    x_ = dataset[t + x_offsets]
    y_ = dataset[t + y_offsets]
    
    if not (np.all(x_ == 0) and np.all(y_ == 0)):
        timeH_x_ = time_h[t + x_offsets]
        timeH_y_ = time_h[t + x_offsets]
        timeM_x_ = time_m[t + x_offsets]
        timeM_y_ = time_m[t + x_offsets]

        x.append(x_)
        y.append(y_)
        timeH_x.append(timeH_x_)
        timeH_y.append(timeH_y_)
        timeM_x.append(timeM_x_)
        timeM_y.append(timeM_y_)

x = np.stack(x, axis=0)
y = np.stack(y, axis=0)
timeH_x = np.stack(timeH_x, axis=0)
timeH_y = np.stack(timeH_y, axis=0)
timeM_x = np.stack(timeM_x, axis=0)
timeM_y = np.stack(timeM_y, axis=0)

num_samples = x.shape[0]
################## split to train, val, test ###########################
num_train = round(num_samples * train_ratio)
num_val = round(num_samples * val_ratio)
num_test = num_samples - num_train - num_val

x_train, y_train = x[:num_train], y[:num_train] 
x_val, y_val = x[num_train: num_train + num_val], y[num_train: num_train + num_val]
x_test, y_test = x[-num_test:], y[-num_test:]

thx_train, thy_train = timeH_x[:num_train], timeH_y[:num_train]  
thx_val, thy_val = timeH_x[num_train: num_train + num_val], timeH_y[num_train: num_train + num_val]
thx_test, thy_test = timeH_x[-num_test:], timeH_y[-num_test:]

tmx_train, tmy_train = timeM_x[:num_train], timeM_y[:num_train]  
tmx_val, tmy_val = timeM_x[num_train: num_train + num_val], timeM_y[num_train: num_train + num_val]
tmx_test, tmy_test = timeM_x[-num_test:], timeM_y[-num_test:]

for cat in ["train", "val", "test"]:
    _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    _thx, _thy = locals()["thx_" + cat], locals()["thy_" + cat]
    _tmx, _tmy = locals()["tmx_" + cat], locals()["tmy_" + cat]
    print(cat, "x: ", _x.shape, "y:", _y.shape)
    np.savez_compressed(
        f'{saved_dir}/{cat}.npz',
        x = _x,
        y = _y,
        x_hour = _thx,
        y_hour = _thy,
        x_minute = _tmx,
        y_minute = _tmy,
    )  
