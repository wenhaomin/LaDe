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
from src.trainers.astgcn_trainer import ASTGCN_Trainer

from src.utils.helper import get_dataloader, check_device, get_num_nodes, get_null_value
from src.utils.metrics import masked_mae
from src.models.astgcn import ASTGCN
from src.trainers.gwnet_trainer import GWNET_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config
from scipy.sparse.linalg import eigs


def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''
    assert W.shape[0] == W.shape[1]
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def get_config():
    parser = get_public_config()

    # get private config
    parser.add_argument('--model_name', type=str, default='astgcn',
                        help='which model to train')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--K', type=int, default=3)

    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.steps = [12000]
    print(args)

    folder_name = '{}-{}-{}-{}-{}-{}'.format(args.n_blocks, args.n_hidden, args.K,
                                             args.aug, args.batch_size, args.base_lr)
    args.log_dir = './logs/{}/{}/{}/'.format(args.dataset,
                                             args.model_name,
                                             folder_name)
    args.num_nodes = get_num_nodes(args.dataset)
    args.null_value = get_null_value(args.dataset)    

    if args.filter_type in ['scalap', 'identity']:
        args.support_len = 1
    else:
        args.support_len = 2

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

    num_nodes = adj_mat.shape[0]
    new_adj = adj_mat + np.eye(num_nodes)
    L_tilde = scaled_Laplacian(new_adj)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(
        device) for i in cheb_polynomial(L_tilde, args.K)]

    model = ASTGCN(nb_block=args.n_blocks,
                   K=args.K,
                   nb_chev_filter=args.n_hidden,
                   nb_time_filter=args.n_hidden,
                   time_strides=1,
                   cheb_polynomials=cheb_polynomials,
                   name=args.model_name,
                   dataset=args.dataset,
                   device=device,
                   num_nodes=args.num_nodes,
                   seq_len=args.seq_len,
                   horizon=args.horizon,
                   input_dim=args.input_dim,
                   output_dim=args.output_dim)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    data = get_dataloader(args.datapath,
                          args.batch_size,
                          args.input_dim,
                          args.output_dim)

    result_path = args.result_path + '/' + args.dataset + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, args.input_dim, args.output_dim)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    trainer = GWNET_Trainer(model=model,
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
                            null_value =args.null_value)

    if args.mode == 'train':
        trainer.train()
        trainer.test(-1, 'test')
    else:
        trainer.test(-1, args.mode)
        if args.save_preds:
            trainer.save_preds(-1)


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()