import torch
import numpy as np
import pandas as pd
import os
import csv
import time
def masked_mse(preds, labels, null_val=np.nan, mask = None):
    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)

    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan, mask = None):
    if mask == None:
        return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))
    else:
        return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val, mask = mask))


def masked_mae(preds, labels, null_val=np.nan, mask = None):

    if mask == None:
        if np.isnan(null_val):
            mask = ~torch.isnan(labels)
        else:
            mask = (labels > null_val + 0.1)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)

    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_mae_dcrnn(preds, labels):
    mask = (labels != 0).float()
    mask /= mask.mean()
    loss = torch.abs(preds - labels)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def compute_all_metrics(pred, real, null_value =np.nan):
    mae = masked_mae(pred, real, null_value).item()
    rmse = masked_rmse(pred, real, null_value).item()
    return mae, rmse

def get_results_csv(labels, preds, null_value, result_path, model_name):

    amae = []
    armse = []

    horizon = preds.shape[1]

    for i in range(horizon):
        pred = preds[:, i]
        real = labels[:, i]
        metrics = compute_all_metrics(pred, real, null_value)
        amae.append(metrics[0])
        armse.append(metrics[1])

    log = 'On average over {} horizons, Average Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(horizon, np.mean(amae), np.mean(armse)))
    
    csv_path  = result_path + '/{}.csv'.format(model_name)
    print('The result path: '.format(csv_path))
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns = ['end_time','mae','rmse'])
        df.to_csv(csv_path, index = False)
        
    with open(csv_path,'a+') as f:
        csv_write = csv.writer(f)
        data_row = [time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()),
                    np.mean(amae), np.mean(armse)]
        csv_write.writerow(data_row)



