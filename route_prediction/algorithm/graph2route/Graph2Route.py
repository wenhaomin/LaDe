# -*- coding: utf-8 -*-

import time
import numpy as np

import torch
import torch.nn as nn

from algorithm.graph2route.gcn import GCNLayer
from algorithm.graph2route.decoder import Decoder

class Graph2Route_pickup(nn.Module):
    def __init__(self, config):
        super(Graph2Route_pickup, self).__init__()
        self.config = config

        self.N = config['max_task_num']  # max nodes
        self.device = config['device']
        # input feature dimension
        self.d_v = config.get('node_fea_dim', 8)  # dimension of node feature
        self.d_e = config.get('edge_fea_dim', 5)  # dimension of edge feature
        self.d_s = config.get('start_fea_dim', 5)  # dimension of start node feature
        self.d_h = config['hidden_size']  # dimension of hidden size
        self.d_w = config.get('worker_emb_dim', 20)  # dimension of worker embedding

        # feature embedding module
        self.worker_emb = nn.Embedding(config['num_worker_logistics'], self.d_w)
        self.node_emb = nn.Linear(self.d_v, self.d_h, bias=False)
        self.edge_emb = nn.Linear(self.d_e, self.d_h, bias=False)
        self.start_node_emb = nn.Linear(self.d_s, self.d_h + self.d_v)#+ self.d_v

        # encoding module
        self.gcn_num_layers = config['gcn_num_layers']
        self.gcn_layers = nn.ModuleList([GCNLayer(hidden_dim=self.d_h, aggregation="mean") for _ in range(self.gcn_num_layers)])
        self.graph_gru = nn.GRU(self.N * self.d_h, self.d_h, batch_first=True)
        self.graph_linear = nn.Linear(self.d_h, self.N * self.d_h)

        # decoding module
        self.decoder = Decoder(
            self.d_h + self.d_v,
            self.d_h + self.d_v,
            self.d_w + 5, # here 5 means the feauture number of courier, you can change it accordingly
            tanh_exploration=10,
            use_tanh=True,
            n_glimpses=1,
            mask_glimpses=True,
            mask_logits=True,
        )


    def dynamic_gnn_encode(self, V, E):
        """
        :param V:  (B,T,N,d_v)
        :param E:  (B,T,N,N,d_e)
        :return:
            Node embeddings: (B,T,N,d_h+d_v)
        """
        B, T, N, d_v = V.shape
        d_h, d_v = self.d_h, self.d_v
        b_node_h = self.node_emb(V) # (B,T,N,d_v) -> (B,T,N,d_h)
        b_edge_h = self.edge_emb(E) # (B,T,N,d_e) -> (B,T,N,d_h)
        # spatial correlation encoding
        for layer in range(self.gcn_num_layers):
            b_node_h, b_edge_h = self.gcn_layers[layer](b_node_h.reshape(B * T, N, d_h), b_edge_h.reshape(B * T, N, N, d_h))

        # temporal correlation encoding
        b_node_h, _ = self.graph_gru(b_node_h.reshape(B, T, -1))
        b_node_h = self.graph_linear(b_node_h)  # （B, T, N * H)

        # concact the feature with original features
        node_H = torch.cat([b_node_h.reshape(B * T, N, d_h), V.reshape(B * T, N, d_v)], dim=2).permute(1, 0, 2).contiguous().clone()

        return node_H


    def personalized_route_decode(self, node_H, D, V_reach_mask, start_fea, cou_fea, start_idx):
        B, T, N = V_reach_mask.shape
        d_h, d_v = self.d_h, self.d_v
        b_decoder_input = torch.zeros([B, T, d_h + d_v]).to(self.device)
        for t in range(T):
            decoder_input = self.start_node_emb(start_fea[:, t, :])
            b_decoder_input[:, t, :] = decoder_input

        cou = torch.repeat_interleave(cou_fea.unsqueeze(1), repeats=T, dim=1).reshape(B * T, -1)  # (B * T, 4)
        cou_id = cou[:, 0].long()
        embed_cou = torch.cat([self.worker_emb(cou_id), cou[:, [1, 2, 3, 4, 5]]], dim=1)  # (B*T, d_w + 5)

        b_init_hx = torch.randn(B * T, d_h + d_v).to(self.device)
        b_init_cx = torch.randn(B * T, d_h + d_v).to(self.device)

        b_V_reach_mask = V_reach_mask.reshape(B * T, N)
        b_inputs = node_H.clone()
        b_enc_h = node_H.clone()
        D_ = D.clone() # (B, T, N, N)
        D_[:, :, :, 0] = 0

        (pointer_log_scores, pointer_argmax, final_step_mask) = \
            self.decoder(
                b_decoder_input.reshape(B * T, d_h + d_v),
                b_inputs.reshape(N, T * B, d_h + d_v),
                (b_init_hx, b_init_cx),
                b_enc_h.reshape(N, T * B, d_h + d_v),
                b_V_reach_mask, embed_cou,  D_.reshape(B * T, N, N), start_idx.reshape(B*T), self.config)
        return pointer_log_scores.exp(), pointer_argmax



    def forward(self, V, V_reach_mask, E, start_fea, start_idx, cou_fea):
        """
        :param V: node features, including [dispatch time, coordinates, relative distance, absolute distance,
                promised time - dispatch time, and promised time - current time]. (B, T, N, d_v), here d_v = 8.
        :param V_reach_mask:  mask for reachability of nodes (B, T, N)
        :param E: masked edge features, include [edge absolute geodesic distance, edge relative geodesic distance, input spatial-temporal adjacent feature,
                                            difference of promised pick-up time between nodes, difference of dispatch_time between nodes] (B, T, N, N, d_e), here d_e = 5
        :param start_fea: features of start nodes at each step, including dispatch time, coordinates, promised pick-up time, finish time (B, T, d_s), here d_s = 5
        :param start_idx: index of start node at each step, (B, T)
        :param cou_fea: features of couriers including id, work days (B, d_w), here d_w = 2

         :return
            pointer_log_scores: (B*T, N, N), the output probability of all nodes at each step
            pointer_argmaxs: (B*T, N), the outputed node at each step
        """

        # Get the node embeddings through the dynmic gnn encoding
        H = self.dynamic_gnn_encode(V, E)

        # Get the distance matrix, which will be used in the decoding process
        D = E[:, :, :, :, 0]  # Distance matrix, D: (B, T, N, N), note that the first feature of E is distance

        # Decoding the route by the personalized route decoder
        pointer_scores, pointer_argmax = self.personalized_route_decode(H, D, V_reach_mask, start_fea, cou_fea, start_idx)

        return pointer_scores, pointer_argmax


    def model_file_name(self):
        t = time.time()
        file_name = '+'.join([f'{k}-{self.config[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.logistics{t}'
        return file_name

# --Dataset
from torch.utils.data import Dataset
class Graph2RouteDataset(Dataset):
    def __init__(
            self,
            mode: str,
            params: dict,
    ) -> None:
        super().__init__()
        if mode not in ["train", "val", "test"]:
            raise ValueError
        path_key = {'train': 'train_path', 'val': 'val_path', 'test': 'test_path'}[mode]
        path = params[path_key]
        self.data = np.load(path, allow_pickle=True).item()

    def __len__(self):

        return len(self.data['V_len'])

    def __getitem__(self, index):

        E_static_fea = self.data['E_static_fea'][index]
        E_mask = self.data['E_mask'][index]
        A = self.data['A'][index]

        V = self.data['V'][index]
        V_len = self.data['V_len'][index]
        V_reach_mask = self.data['V_reach_mask'][index]

        label = self.data['route_label'][index]
        label_len = self.data['label_len'][index]
        start_fea = self.data['start_fea'][index]
        start_idx = self.data['start_idx'][index]
        cou_fea = self.data['cou_fea'][index]

        return V, V_len, V_reach_mask, E_static_fea, E_mask, A, start_fea, start_idx, cou_fea, label, label_len

# ---Log--
from utils.util import ws, save2file_meta
def save2file(params):
    # from utils.util import ws
    file_name = ws + f'/output/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size','k_nearest_neighbors', 'long_loss_weight',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',

    ]
    save2file_meta(params,file_name,head)