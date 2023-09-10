# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from algorithm.ranketpa.transformer import TransformerEncoder


class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=True):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1)) #输出N个节点的eta
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x.float())


def rnn_forwarder(rnn, embedded_inputs, input_lengths, batch_size):
    """
    :param embedded_inputs:
    :param input_lengths:
    :param batch_size:
    :param rnn: RNN instance
    :return: the result of rnn layer,
    """
    packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths.cpu(),
                                               batch_first=rnn.batch_first, enforce_sorted=False)

    # Forward pass through RNN
    try:
        outputs, hidden = rnn(packed)
    except:
        print('lstm encoder:', embedded_inputs)

    # Unpack padding
    outputs, index = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=rnn.batch_first)

    # Return output and final hidden state
    if rnn.bidirectional:
        # Optionally, Sum bidirectional RNN outputs
        outputs = outputs[:, :, :rnn.hidden_size] + outputs[:, :, rnn.hidden_size:]

    index2 = index - torch.tensor([1])  # 选取操作
    index1 = torch.tensor(list(range(len(index2))))  # 生成第一维度
    return outputs[index1, index2, :]



def get_sinusoid_encoding(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

class RankEPTA(nn.Module):
    def __init__(self, args):
        super(RankEPTA, self).__init__()
        self.args = args
        self.max_len = args['max_task_num']

        # last_x_size = args['last_x_size']
        self.sort_x_size = args['sort_x_size']
        self.embed_dim = 20

        self.emb_dim = args['hidden_size']
        self.hidden_size = args['hidden_size']
        self.number_layer = 2
        self.n_head = 8

        self.pos_table = get_sinusoid_encoding(26, self.embed_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pos_table = self.pos_table.to(self.device)
        # for sort_x embedding layer
        self.sort_x_embedding = nn.Linear(in_features=self.sort_x_size, out_features=self.hidden_size, bias=False)
        self.mlp_eta = MLP(self.embed_dim  + args['hidden_size'] * 2 , (self.hidden_size,), dropout=0)

        self.todo_emb = nn.Sequential(nn.Linear(self.sort_x_size, self.sort_x_size//2),
                                      nn.LeakyReLU(),
                                      nn.Linear(self.sort_x_size //2, 1),
                                      nn.LeakyReLU()
                                      )


        # STattention-based PETA predictor
        self.transformer = TransformerEncoder(n_heads=self.n_head, node_dim=self.hidden_size+self.embed_dim,
                                              embed_dim=self.hidden_size, n_layers=self.number_layer, normalization='batch')

        #prediction
        self.linear_eta = nn.Sequential(nn.Linear(in_features=2 * self.hidden_size + self.max_len + 1 + self.emb_dim,
                                                  out_features=(self.hidden_size + self.max_len + self.hidden_size) // 2, bias=False), nn.ELU(),
                                        nn.Linear(in_features=(self.hidden_size + self.max_len + self.hidden_size) // 2, out_features=32, bias=False), nn.ELU(),
                                        nn.Linear(in_features=32, out_features=1, bias=False), nn.ReLU())

    def get_att_mask(self, x):
        B, N = x.shape
        att_mask = torch.zeros(B, N, N)
        x = x.unsqueeze(-1)
        for i in range(B):
            att_mask[i] = x[i] * x[i].T
        return att_mask


    def forward(self, V, V_reach_mask, sort_pos):
        B = V_reach_mask.size(0)
        T = V_reach_mask.size(1)
        N = V_reach_mask.size(2)

        order_info = sort_pos
        order_info = self.pos_table[order_info.long()].float()
        mask_index = V.reshape(-1, N, V.shape[-1])[:, :, 0] == 0
        attn_mask = (~mask_index + 0).repeat_interleave(25).reshape(B*T, N, N)
        sort_x_emb = self.sort_x_embedding(V.reshape(B*T, N, -1).float())  # (batch_size, max_seq_len, todo_emb_dim)
        x = torch.cat([sort_x_emb, order_info.reshape(B*T,N,-1)], dim = 2)
        transformer_output, _ = self.transformer(x, attn_mask)

        # ETA final prediction
        current_state = sort_x_emb
        F_input = torch.cat([transformer_output, current_state, order_info.reshape(B*T,N,-1)], dim=2)
        eta = self.mlp_eta(F_input)

        return eta


    def model_file_name(self):
        file_name = '+'.join([f'{k}${self.args[k]}' for k in [ 'hidden_size', 'dataset']])
        file_name = f'{file_name}.rankpeta'
        return file_name

from utils.util import save2file_meta, ws
def save2file(params):
    file_name = ws + f'/output/time_prediction/{params["model"]}.csv'
    # 写表头
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop',  'is_test', 'log_time',
        # metric result

        'acc_eta@10', 'acc_eta@20', 'acc_eta@30', 'acc_eta@40', 'acc_eta@50', 'acc_eta@60','rmse', 'mae',
    ]
    save2file_meta(params,file_name,head)
