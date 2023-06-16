import os
import sys
from os.path import join
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import time
import argparse
import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from src.trainers.stgncde_trainer import Trainer
from src.utils.helper_stgncde import get_dataloader_cde, init_seed, make_model

torch.set_num_threads(3)

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--device', default='cuda:0', type=str)
args.add_argument('--dataset_name', default='Delivery_SH', type=str)
args.add_argument('--seed', default=2018, type=int)
args.add_argument('--n_exp', default=0, type=int)

args.add_argument('--embed_dim', default=10, type=int)
args.add_argument('--hid_dim', default=32, type=int)
args.add_argument('--hid_hid_dim', default=64, type=int)
args.add_argument('--num_layers', default=2, type=int)
args.add_argument('--cheb_k', default=2, type=int)
args.add_argument('--solver', default='rk4', type=str)

args.add_argument('--batch_size', default=16, type=int)
args.add_argument('--epochs', default=100, type=int)
args.add_argument('--patience', default=10, type=int)
args.add_argument('--lr_init', default=1e-3, type=float)
args.add_argument('--weight_decay', default=1e-3, type=eval)
args.add_argument('--lr_decay', default=True, type=eval)
args.add_argument('--lr_decay_rate', default=0.3, type=float)
args.add_argument('--lr_decay_step', default='30,70,90', type=str)
args.add_argument('--early_stop', default=True, type=eval)
args.add_argument('--early_stop_patience', default=20, type=int)
args.add_argument('--grad_norm', default=False, type=eval)
args.add_argument('--max_grad_norm', default=5, type=int)


args.add_argument('--model', default='GCDE', type=str)
args.add_argument('--mode', default='train', type=str)
args.add_argument('--debug', default=False, type=eval)
args.add_argument('--model_type', default='type1', type=str)
args.add_argument('--g_type', default='agc', type=str)
args.add_argument('--input_dim', default=2, type=int)
args.add_argument('--output_dim', default=1, type=int)
args.add_argument('--val_ratio', default=0.2, type=float)
args.add_argument('--test_ratio', default=0.2, type=float)
args.add_argument('--seq_len', default=24, type=int)
args.add_argument('--horizon', default=24, type=int)
args.add_argument('--num_nodes', default=0, type=int)
args.add_argument('--tod', default=False, type=eval)
args.add_argument('--normalizer', default='std', type=str)
args.add_argument('--column_wise', default=False, type=eval)
args.add_argument('--default_graph', default=True, type=eval)
args.add_argument('--loss_func', default='mae', type=str)
args.add_argument('--teacher_forcing', default=False, type=bool)
args.add_argument('--real_value', default=True, type=eval, help = 'use real value for loss calculation')
args.add_argument('--missing_test', default=False, type=bool)
args.add_argument('--missing_rate', default=0.1, type=float)
args.add_argument('--mae_thresh', default=None, type=eval)
args.add_argument('--mape_thresh', default=0., type=float)
args.add_argument('--model_path', default='', type=str)
# args.add_argument('--log_dir', default='../runs', type=str)
args.add_argument('--log_step', default=0, type=int)
args.add_argument('--plot', default=False, type=eval)
args.add_argument('--tensorboard',action='store_true',help='tensorboard')
args = args.parse_args()
print(args)

if args.dataset_name == 'Delivery_SH':
    args.dataset = './data/Delivery_SH'
    args.num_nodes = 30
    args.input_dim = 2
    args.output_dim = 1
    null_value = -1.0

elif args.dataset_name == 'Delivery_HZ':
    args.dataset = './data/Delivery_HZ'
    args.num_nodes = 31
    args.input_dim = 2
    args.output_dim = 1
    null_value = -1.0

elif args.dataset_name == 'Delivery_CQ':
    args.dataset = './data/Delivery_CQ'
    args.num_nodes = 30
    args.input_dim = 2
    args.output_dim = 1
    null_value = -1.0

init_seed(args.seed)

device = torch.device(args.device)

#config log path
args.log_dir = './logs/{}/stgncde/'.format(args.dataset_name)
if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
print(args.log_dir)

# if args.tensorboard:
#     w : SummaryWriter = SummaryWriter(args.log_dir)
# else:
#     w = None

#init model
if args.model_type=='type1':
    model, vector_field_f, vector_field_g = make_model(args)
elif args.model_type=='type1_temporal':
    model, vector_field_f = make_model(args)
elif args.model_type=='type1_spatial':
    model, vector_field_g = make_model(args)
else:
    raise ValueError("Check args.model_type")

model = model.to(device)

if args.model_type=='type1_temporal':
    vector_field_f = vector_field_f.to(device)
    vector_field_g = None
elif args.model_type=='type1_spatial':
    vector_field_f = None
    vector_field_g = vector_field_g.to(device)
else:
    vector_field_f = vector_field_f.to(device)
    vector_field_g = vector_field_g.to(device)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    else:
        nn.init.uniform_(p)

#load dataset
train_loader, val_loader, test_loader, scaler, times = get_dataloader_cde(args, normalizer=args.normalizer, tod=args.tod, dow=False, weather=False, single=False)

#init loss function, optimizer
loss = None

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, weight_decay=args.weight_decay)

#learning rate decay
lr_scheduler = None
if args.lr_decay:
    print('Applying learning rate decay.')
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=args.patience, verbose=True)

result_path = './results/' + args.dataset_name + '/{}_{}_{}_{}'.format(args.seq_len, args.horizon, int(args.input_dim - 1), args.output_dim)

#start training
trainer = Trainer(model, vector_field_f, vector_field_g, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler, device, times, result_path, null_value,args.dataset_name)
if args.mode == 'train':
    trainer.train()
    trainer.test(model, trainer.args, test_loader, times)
elif args.mode == 'test':
    # model.load_state_dict(torch.load('./pre-trained/{}.pth'.format(args.dataset_name)))
    # print("Load saved model")
    trainer.test(model, trainer.args, test_loader,  times)
else:
    raise ValueError
