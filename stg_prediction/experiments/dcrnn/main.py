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
from src.models.dcrnn import DCRNNModel
from src.trainers.dcrnn_trainer import DCRNN_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config


def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='dcrnn',
                        help='which model to train')
    parser.add_argument('--filter_type', type=str, default='dual_random_walk')
    parser.add_argument('--cl_decay_steps', type=int, default=2000)
    parser.add_argument('--l1_decay', type=int, default=0)
    parser.add_argument('--max_diffusion_step', type=int, default=2)
    parser.add_argument('--num_rnn_layers', type=int, default=2)
    parser.add_argument('--rnn_units', type=int, default=32)

    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.steps = [10, 70, 110]
    print(args)

    folder_name = '{}-{}-{}'.format(args.num_rnn_layers, args.rnn_units, args.aug)
    args.log_dir = './logs/{}/{}/{}/'.format(args.dataset,
                                             args.model_name,
                                             folder_name)
    args.num_nodes = get_num_nodes(args.dataset)
    args.null_value = get_null_value(args.dataset)

    args.datapath = os.path.join('./data', args.dataset)
    args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower())
    if args.seed != 0:
        torch.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args


def main():
    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)

    model = DCRNNModel(max_diffusion_step=args.max_diffusion_step,
                       num_rnn_layers=args.num_rnn_layers,
                       n_filters=args.rnn_units,
                       filter_type=args.filter_type,
                       cl_decay_steps=args.cl_decay_steps,
                       use_curriculum_learning=True,
                       name=args.model_name,
                       dataset=args.dataset,
                       device=device,
                       num_nodes=args.num_nodes,
                       seq_len=args.seq_len,
                       horizon=args.horizon,
                       input_dim=args.input_dim,
                       output_dim=args.output_dim)

    data = get_dataloader(args.datapath,
                          args.batch_size,
                          args.input_dim,
                          args.output_dim)

    result_path = args.result_path + '/' + args.dataset + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    trainer = DCRNN_Trainer(model=model,
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
                            device=device,
                            result_path=result_path,                  
                            model_name=args.model_name,
                            null_value =args.null_value, 
                            )

    if args.mode == 'train':
        trainer.train()
        trainer.test(-1, 'test')
    else:
        trainer.test(-1, args.mode)
        if args.save_preds:
            trainer.save_preds(-1)


if __name__ == "__main__":
    main()