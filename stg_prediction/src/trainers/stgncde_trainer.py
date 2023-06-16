import torch
import math
import os
import time
import copy
import numpy as np
import src.utils.metrics as mc
from src.utils.logging import get_logger
from src.utils.helper_stgncde import print_model_parameters
from torch import nn, Tensor
from typing import Optional, List, Union

class Trainer(object):
    def __init__(self, model, vector_field_f, vector_field_g, loss, optimizer, train_loader, val_loader, test_loader, scaler, args, lr_scheduler, device, times, result_path, null_value, dataset_name):
        super(Trainer, self).__init__()
        self.model = model
        self.vector_field_f = vector_field_f 
        self.vector_field_g = vector_field_g
        self.loss = mc.masked_mae
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.logger = get_logger(args.log_dir, name=args.model)
        total_param = print_model_parameters(model, only_num=False)
        self.logger.info("Total params: {}".format(str(total_param)))
        self.device = device
        self.times = times.to(self.device, dtype=torch.float)
        self.result_path = result_path
        self.null_value = null_value
        self._save_path = args.log_dir
        self._n_exp = args.n_exp

    def save_model(self):
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
        filename = 'final_model_{}.pt'.format(self._n_exp)
        torch.save(self.model.state_dict(), os.path.join(self._save_path, filename))


    def load_model(self):
        filename = 'final_model_{}.pt'.format(self._n_exp)
        self.model.load_state_dict(torch.load(
            os.path.join(self._save_path, filename)))


    def inverse_transform(self, tensors: Union[Tensor, List[Tensor]]):
        def inv(tensor, scalers):
            for i in range(tensor.shape[2]):
                tensor[:,:, i,:1] = scalers[i].inverse_transform(tensor[:,:, i,:1])
            return tensor

        if isinstance(tensors, list):
            return [inv(tensor, self.scaler) for tensor in tensors]
        else:
            return inv(tensors, self.scaler)

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0
        total_rmse = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
                *valid_coeffs, target = batch
                label = target[..., :self.args.output_dim]
                output = self.model(self.times, valid_coeffs)
                if self.args.real_value:     
                    label = self.inverse_transform(label)

                real_min = 0.0
                if torch.min(label) < 1:
                    real_min = torch.min(label).cpu()

                loss = self.loss(output, label, self.null_value)
                rmse = mc.masked_rmse(output, label, self.null_value).item()
    
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    total_rmse += rmse
        
        val_loss = total_val_loss / len(val_dataloader)
        val_rmse = total_rmse / len(val_dataloader)
        self.logger.info('Epoch {}, Valid MAE: {:.4f}, Valid RMSE: {:.4f}'.format(epoch, val_loss, val_rmse))
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_rmse = 0
        for batch_idx, batch in enumerate(self.train_loader):
            batch = tuple(b.to(self.device, dtype=torch.float) for b in batch)
            *train_coeffs, target = batch
            label = target[..., :self.args.output_dim]  # (..., 1)
            self.optimizer.zero_grad()
            
            output = self.model(self.times, train_coeffs)
            
            if self.args.real_value:
                label = self.inverse_transform(label)

            
            real_min = 0.0
            if torch.min(label) < 1:
                real_min = torch.min(label).cpu()
            
            loss = self.loss(output, label, self.null_value)
            rmse = mc.masked_rmse(output, label, self.null_value).item()
            
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()
            total_rmse += rmse
                
        train_epoch_loss = total_loss / self.train_per_epoch
        train_epoch_rmse = total_rmse / self.train_per_epoch
        self.logger.info('Epoch {}, Train MAE: {:.4f}, Train RMSE: {:.4f}'.format(epoch, train_epoch_loss, train_epoch_rmse))

        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            
            train_start = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            train_end = time.time()
            
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            
            val_start = time.time()
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            val_end = time.time()

            self.logger.info('Epoch {}, Train Time: {:.4f}, Val Time: {:.4f}'.format(epoch, train_end-train_start, val_end-val_start))
            
            #learning rate decay
            if self.args.lr_decay:
                self.lr_scheduler.step(val_epoch_loss)
            
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break

            # save the best state
            if best_state == True:
                self.logger.info('Start testing')
                self.test_simple(self.model, self.args, self.test_loader, self.scaler, self.logger, None, self.times)
                self.save_model()
                

    def test_simple(self, model, args, data_loader, scaler, logger, path, times):
        sta_time = time.time()
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
                *test_coeffs, target = batch
                label = target[..., :args.output_dim]
                output = model(times.to(args.device, dtype=torch.float), test_coeffs)
                y_true.append(label)
                y_pred.append(output)
        y_true = self.inverse_transform(torch.cat(y_true, dim=0))

        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = self.inverse_transform(torch.cat(y_pred, dim=0))

        metrics = mc.compute_all_metrics(y_pred, y_true, self.null_value)

        end_time = time.time()
        log = 'Average Test MAE: {:.4f}, Test RMSE: {:.4f}, test_time: {:.1f}'
        self.logger.info(log.format(metrics[0], metrics[1], (end_time-sta_time)))

    def test(self, model, args, data_loader, times):
        # load model
        self.model = model
        self.load_model()
        self.model.eval()

        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(b.to(args.device, dtype=torch.float) for b in batch)
                *test_coeffs, target = batch
                label = target[..., :args.output_dim]
                output = self.model(times.to(args.device, dtype=torch.float), test_coeffs)
                y_true.append(label)
                y_pred.append(output)
        y_true = self.inverse_transform(torch.cat(y_true, dim=0))

        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = self.inverse_transform(torch.cat(y_pred, dim=0))

        mc.get_results_csv(y_true, y_pred, self.null_value, self.result_path, 'stgncde')
            
        # return np.mean(amae)

