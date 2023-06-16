import numpy as np
import src.utils.metrics as mc
import torch
import pandas as pd
import os

dataset_name = 'Delivery_SH'
base_path = './data/{}/'.format(dataset_name)

seq_len = 24
horizon = 24
in_dim = 1
out_dim = 1
nan_value = -1
result_path = './results/' + dataset_name + '/{}_{}_{}_{}'.format(seq_len, horizon, in_dim, out_dim)
model_name = 'ha'

train_x = np.load(base_path + 'train.npz')['x'][..., :in_dim]
val_x = np.load(base_path + 'val.npz')['x'][..., :in_dim]
test_y = np.load(base_path + 'test.npz')['y'][..., :out_dim]

train_ha = np.concatenate([train_x,val_x])
test_ha = test_y

print('##### Train and test set for HA #####')
for i in [train_ha, test_ha]:
    print(i.shape)

train_x_hour = np.load(base_path + 'train.npz')['x_hour']
val_x_hour = np.load(base_path + 'val.npz')['x_hour']
test_y_hour = np.load(base_path + 'test.npz')['y_hour']

train_ha_hour = np.concatenate([train_x_hour,val_x_hour])
test_ha_hour = test_y_hour

##### get the average of each hour #####
n_sample, n_time, n_node, _ = train_ha.shape
ave_ha = np.zeros((24,n_node))
train_ha = train_ha.reshape(n_sample*n_time, n_node)
train_ha_hour = train_ha_hour.reshape(n_sample*n_time)
for h in range(24):
    mask = [train_ha_hour == h]
    mean_train = np.mean(train_ha[train_ha_hour == h], 0)
    ave_ha[h] = mean_train

##### do the prediction #######
pred_y = np.zeros_like(test_ha)
pred_y = pred_y.reshape(-1, n_node)
test_ha_hour = test_ha_hour.reshape(-1)
for h in range(24):
    pred_y[test_ha_hour == h] = ave_ha[h]
pred_y=pred_y.reshape(test_ha.shape)

#### get the metrics ####
if not os.path.exists(result_path):os.makedirs(result_path)
mc.get_results_csv(torch.Tensor(test_ha), torch.Tensor(pred_y), nan_value, result_path, model_name)

