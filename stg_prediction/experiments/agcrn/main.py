import os
import time
import random
import torch
import argparse
import numpy as np

from src.utils.helper import get_dataloader, check_device, get_num_nodes, get_null_value
from src.utils.metrics import masked_mae
from src.models.agcrn import AGCRN
from src.trainers.agcrn_trainer import AGCRN_Trainer
from src.utils.graph_algo import load_graph_data
from src.utils.args import get_public_config

def get_config():
    parser = get_public_config()
    parser.add_argument('--model_name', type=str, default='agcrn',help='which model to train')
    parser.add_argument('--filter_type', type=str, default='doubletransition')
    parser.add_argument('--embed_dim', type=int, default=10, help='embed dimension')
    parser.add_argument('--rnn_units', type=int, default=32, help='number of hidden dimensions')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    parser.add_argument('--cheb_k', type=int, default=2, help='cheb order')

    parser.add_argument('--seed',type=int, default=0, help='random seed')
    args = parser.parse_args()
    print(args) 
    args.steps = [12000]
    folder_name = '{}-{}-{}-{}-{}-{}'.format(args.num_layers, args.num_layers, args.cheb_k,
                                            args.embed_dim, args.batch_size, args.base_lr)
    args.log_dir = './logs/{}/{}/{}/'.format(args.dataset,
                                                args.model_name,
                                                folder_name)

    if args.filter_type in ['scalap', 'identity']:
        args.support_len = 1
    else:
        args.support_len = 2

    args.datapath = os.path.join('./data', args.dataset)
    args.graph_pkl = 'data/sensor_graph/adj_mx_{}.pkl'.format(args.dataset.lower())
    args.num_nodes = get_num_nodes(args.dataset)
    args.null_value = get_null_value(args.dataset) 

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return args


def main():
    args = get_config()
    device = check_device()
    _, _, adj_mat = load_graph_data(args.graph_pkl)

    if args.seed != 0:
        torch.manual_seed(args.seed)

    model = AGCRN(embed_dim = args.embed_dim, 
                    rnn_units=args.rnn_units, 
                    num_layers=args.num_layers, 
                    cheb_k=args.cheb_k,
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

    trainer = AGCRN_Trainer(model=model,
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