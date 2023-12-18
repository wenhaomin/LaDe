# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=True):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 25)) #输出N个节点的eta
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x.float())

class MLP_ETA(nn.Module):
    def __init__(self, args={}):
        super(MLP_ETA, self).__init__()

        # network parameters
        self.hidden_size = args['hidden_size']
        self.args = args
        self.mlp_eta = MLP(args['max_task_num'] * 8 + 4, (self.hidden_size,))

    def forward(self, V, V_reach_mask, start_fea, cou_fea):

        B =  V_reach_mask.size(0)
        T =  V_reach_mask.size(1)
        N =  V_reach_mask.size(2)
        mask_index = V.reshape(-1, N, V.shape[-1])[:, :, 0] == 0 #[B, T, N]
        V_dis = V.reshape(B*T, N, -1)[:, :, [-2, -1]]
        cou_speed = cou_fea.unsqueeze(1).repeat(1, T, 1).reshape(B * T, -1)[:, -1]
        V_avg_t = V_dis / cou_speed.unsqueeze(1).unsqueeze(1).repeat(1, N, 1)
        mask = (~mask_index +0).reshape(B*T, N, 1)
        eta_input = torch.cat([V.reshape(B*T, N, -1), V_avg_t.reshape(B*T, N, -1)], dim=2) * mask
        eta_input = torch.cat([eta_input.reshape(B * T, -1), start_fea.reshape(B*T, -1)], dim = 1)
        eta = self.mlp_eta(eta_input)

        return eta

    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.args[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.mlp{time.time()}.csv'
        return file_name

# -------------------------------------------------------------------------------------------------------------------------#

from utils.util import save2file_meta
def save2file(params):
    from utils.util import save2file_meta, ws
    file_name = ws + f'/output/time_prediction_1215/{params["dataset"]}/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'task', 'eval_min', 'eval_max',
        # model parameters
        'model',
        # training set
        'num_epoch', 'batch_size', 'is_test', 'log_time',
        # metric result
        'acc_eta@10', 'acc_eta@20', 'acc_eta@30', 'acc_eta@40', 'acc_eta@50', 'acc_eta@60','mae', 'rmse',
    ]
    save2file_meta(params, file_name, head)
