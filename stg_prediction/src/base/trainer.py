import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from src.utils.logging import get_logger
from src.utils import metrics as mc
from src.utils.metrics import masked_mae
from src.base.sampler import RandomSampler, CutEdgeSampler

import pandas as pd
import csv


class BaseTrainer():
    def __init__(
            self,
            model: nn.Module,
            adj_mat,
            filter_type: str,
            data,
            aug: float,
            base_lr: float,
            steps,
            lr_decay_ratio,
            log_dir: str,
            n_exp: int,
            save_iter: int = 300,
            clip_grad_value: Optional[float] = None,
            max_epochs: Optional[int] = 1000,
            patience: Optional[int] = 1000,
            device: Optional[Union[torch.device, str]] = None,
            model_name: str = None,
            result_path: str = None,
            hp: str = None,
            null_value: float = 0.0,
    ):
        super().__init__()

        if n_exp == None:
            n_exp = time.time()

        self._logger = get_logger(
            log_dir, __name__, 'info_{}.log'.format(n_exp), level=logging.INFO)
        if device is None:
            print("`device` is missing, try to train and evaluate the model on default device.")
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)

        self._model = model
        self.model.to(self._device)
        self._logger.info("the number of parameters: {}".format(self.model.param_num(self.model.name))) 

        self._adj_mat = adj_mat
        self._filter_type = filter_type
        self._aug = aug
        self._loss_fn = masked_mae
        # self._loss_fn.to(self._device)
        self._base_lr = base_lr
        self._optimizer = Adam(self.model.parameters(), base_lr)
        self._lr_decay_ratio = lr_decay_ratio
        self._steps = steps
        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(self.optimizer,
                                             steps,
                                             gamma=lr_decay_ratio)
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        if result_path != None:
            self._result_path = result_path
        else: 
            self._result_path = './results'
        self._n_exp = n_exp
        self._data = data
        self._supports = None
        self.model_name = model_name
        self.hp = hp
        self.null_value = null_value
        
        if aug > 0:
            self._sampler = RandomSampler(adj_mat, filter_type)
            # self._sampler = CutEdgeSampler(adj_mat, filter_type, 200)
        
        self._supports = self._calculate_supports(adj_mat, filter_type)
        assert(self._supports is not None)

    @property
    def model(self):
        return self._model

    @property
    def supports(self):
        return self._supports

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path
    
    @property
    def result_path(self):
        return self._result_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            try: 
                return [tensor.to(self._device) for tensor in tensors]
            except AttributeError:
                x, label = tensors
                x = [tensor.to(self._device) for tensor in x]
                label = label.to(self._device)
                return [x, label]
        else:
            return tensors.to(self._device)

    def _inverse_transform(self, tensors: Union[Tensor, List[Tensor]]):
        def inv(tensor, scalers):
            for i in range(tensor.shape[2]):
                tensor[:,:, i,:1] = scalers[i].inverse_transform(tensor[:,:, i,:1])
            return tensor

        if isinstance(tensors, list):
            return [inv(tensor, self.data['scalers']) for tensor in tensors]
        else:
            return inv(tensors, self.data['scalers'])


    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_{}.pt'.format(n_exp)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return True

    def load_model(self, epoch, save_path, n_exp):
        filename = 'final_model_{}.pt'.format(n_exp)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
        return True

    def early_stop(self, epoch, best_loss):
        self.logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch, best_loss))
        np.savetxt(os.path.join(self.save_path, 'val_loss_{}.txt'.format(self._n_exp)), [best_loss], fmt='%.4f', delimiter=',')

    def _calculate_supports(self, adj_mat, filter_type):
        return None

    def train_batch(self, X, label, iter):
        if self._aug < 1:
            new_adj = self._sampler.sample(self._aug)
            supports = self._calculate_supports(new_adj, self._filter_type)
        else:
            supports = self.supports
        self.optimizer.zero_grad()
        pred = self.model(X, supports)
        pred, label = self._inverse_transform([pred, label])

        loss = self.loss_fn(pred, label, self.null_value )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")
        # training phase
        iter = 0
        val_losses = [np.inf]
        saved_epoch = -1
        for epoch in range(self._max_epochs):
            self.model.train()
            train_losses = []
            if epoch - saved_epoch > self._patience:
                self.early_stop(epoch, min(val_losses))
                break

            start_time = time.time()
            for i, data in enumerate(self.data['train_loader']):
                (X, label) = data
                X, label = self._check_device([X, label])
                train_losses.append(self.train_batch(X, label, iter))
                iter += 1
                if iter != None:
                    if iter % self._save_iter == 0:
                        val_loss = self.evaluate()
                        message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f} '.format(epoch,
                                    self._max_epochs,
                                    iter,
                                    np.mean(train_losses),
                                    val_loss)
                        self.logger.info(message)

                        if val_loss < np.min(val_losses):
                            model_file_name = self.save_model(
                                epoch, self._save_path, self._n_exp)
                            self._logger.info(
                                'Val loss decrease from {:.4f} to {:.4f}, '
                                'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                            val_losses.append(val_loss)
                            saved_epoch = epoch
                            
            end_time = time.time()
            self.logger.info("epoch complete")
            self.logger.info("evaluating now!")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            val_loss = self.evaluate()

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                '{:.1f}s'.format(epoch,
                                 self._max_epochs,
                                 iter,
                                 np.mean(train_losses),
                                 val_loss,
                                 new_lr,
                                 (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

    def evaluate(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for data in self.data['val_loader']:
                (X, label) = data
                X, label = self._check_device([X, label])
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())
           
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
        mae = self.loss_fn(preds, labels, self.null_value ).item()
        return mae

    def test_batch(self, X, label):
        pred = self.model(X, self.supports)
        pred, label = self._inverse_transform([pred, label])
        return pred, label

    def test(self, epoch, mode='test'):
        self.load_model(epoch, self.save_path, self._n_exp)

        labels = []
        preds = []
        
        start_time = time.time()
        
        with torch.no_grad():
            self.model.eval()
            for _, data in enumerate(self.data[mode + '_loader']):
                (X, label) = data
                X, label = self._check_device([X, label])
                pred, label = self.test_batch(X, label)
                labels.append(label.cpu())
                preds.append(pred.cpu())

        end_time = time.time()
        
        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)
         
        amae = []
        armse = []

        for i in range(self.model.horizon):
            pred = preds[:, i]
            real = labels[:, i]
            metrics = mc.compute_all_metrics(pred, real, self.null_value )
            log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, teat_time {:.1f}s'
            print(log.format(i+1, metrics[0], metrics[1], (end_time - start_time)))
            amae.append(metrics[0])
            armse.append(metrics[1])

        log = 'On average over {} horizons, Average Test MAE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(self.model.horizon, np.mean(amae), np.mean(armse)))
        
 
        csv_path  = self.result_path + '/{}.csv'.format(self.model_name)
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns = ['hp','end_time','time',
                                            'mae','rmse'])
            df.to_csv(csv_path, index = False)
            
        with open(csv_path,'a+') as f:
            csv_write = csv.writer(f)
            data_row = [self.hp, time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), round(end_time - start_time, 2),
                        np.mean(amae), np.mean(armse)]
            csv_write.writerow(data_row)
            
        return np.mean(amae)


    def save_preds(self, epoch):
        self.load_model(epoch, self.save_path, self._n_exp)

        for mode in ['train', 'val', 'test']:
            labels = []
            preds = []
            with torch.no_grad():
                self.model.eval()
                for _, (X, label) in enumerate(self.data[mode + '_loader']):
                    X, label = self._check_device([X, label])
                    pred, label = self.test_batch(X, label)
                    labels.append(label.cpu())
                    preds.append(pred.cpu())
            labels = torch.cat(labels, dim=0)
            preds = torch.cat(preds, dim=0)

            np.save(os.path.join(self.save_path, mode + '_preds.npy'), preds)
            np.save(os.path.join(self.save_path, mode + '_labels.npy'), labels)
