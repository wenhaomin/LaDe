import torch
import numpy as np
import os
import time
import argparse
import yaml
import pickle
import scipy.sparse as sp
from scipy.sparse import linalg

import torch.nn as nn
import torch

from src.utils.helper import get_dataloader, check_device, get_num_nodes, get_null_value
from src.utils.metrics import masked_mae
from src.models.mtgnn import MTGNN
from src.trainers.mtgnn_trainer import MTGNN_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config, str_to_bool

def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='mtgnn',
                        help='which model to train')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--residual_channels', type=int, default=32)
    parser.add_argument('--dilation_channels', type=int, default=32)

    parser.add_argument('--gcn_true', type=str_to_bool, default=True, help='whether to add graph convolution layer')
    parser.add_argument('--buildA_true', type=str_to_bool, default=True,help='whether to construct adaptive adjacency matrix')
    parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
    parser.add_argument('--subgraph_size',type=int,default=10,help='k')
    parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
    parser.add_argument('--dilation_exponential',type=int,default=1,help='dilation exponential')

    parser.add_argument('--conv_channels',type=int,default=32,help='convolution channels')
    parser.add_argument('--skip_channels',type=int,default=64,help='skip channels')
    parser.add_argument('--end_channels',type=int,default=128,help='end channels')

    parser.add_argument('--layers',type=int,default=3,help='number of layers')

    parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
    parser.add_argument('--tanhalpha',type=float,default=3,help='adj alpha')
    parser.add_argument('--seed',type=int,default=0)

    args = parser.parse_args()
    args.steps = [12000]
    print(args)

    folder_name = '{}-{}-{}-{}-{}'.format(args.gcn_depth, args.conv_channels, args.layers,args.node_dim,args.buildA_true)
    args.log_dir = './logs/{}/{}/{}/'.format(args.dataset,
                                             args.model_name,
                                             folder_name)
    args.num_nodes = get_num_nodes(args.dataset)
    args.null_value = get_null_value(args.dataset)                                             

    
    if args.filter_type == 'scalap':
        args.support_len = 1
    else:
        args.support_len = 2

    args.datapath = os.path.join('./data', args.dataset)
    args.adj_data = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower())
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args


def main():

    args = get_config()
    torch.set_num_threads(3)

    device = check_device()
    _, _, adj_mat = load_graph_data(args.adj_data)

    model = MTGNN(gcn_true=args.gcn_true, 
                  buildA_true=args.buildA_true, 
                  gcn_depth=args.gcn_depth, 
                  device=device, 
                  predefined_A=adj_mat,
                  dropout=args.dropout, 
                  subgraph_size=args.subgraph_size,
                  node_dim=args.node_dim,
                  dilation_exponential=args.dilation_exponential,
                  conv_channels=args.conv_channels, 
                  residual_channels=args.residual_channels,
                  skip_channels=args.skip_channels, 
                  end_channels= args.end_channels,
                  layers=args.layers, 
                  propalpha=args.propalpha, 
                  tanhalpha=args.tanhalpha, 
                  layer_norm_affline=True,
                  dataset=args.dataset,
                  name=args.model_name,
                  num_nodes=args.num_nodes,
                  seq_len=args.seq_len,
                  horizon=args.horizon,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim
                  )

    data = get_dataloader(args.datapath,
                          args.batch_size,
                          args.input_dim,
                          args.output_dim)
    result_path = args.result_path + '/' + args.dataset + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    trainer = MTGNN_Trainer(model=model,
                            adj_mat=adj_mat,
                            filter_type=args.filter_type,
                            data=data,
                            aug=args.aug,
                            base_lr=args.base_lr,
                            steps=args.steps,
                            lr_decay_ratio=args.lr_decay_ratio,
                            log_dir=args.log_dir,
                            n_exp=args.n_exp,
                            save_iter=args.save_iter,
                            clip_grad_value=args.max_grad_norm,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            model_name=args.model_name,
                            result_path=result_path,
                            null_value =args.null_value,
                            device=device
                            )

    if args.mode == 'train':
        trainer.train()
        trainer.test(-1, 'test')
    else:
        trainer.test(-1, args.mode)

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()