from data_provider.data_factory import *
from exp.exp_basic import Exp_Basic
from models import (Informer, Transformer, PatchTST, TCNTorch, LLMPatchTST)
from utils.tools import EarlyStopping, adjust_learning_rate, visual, get_model_size
from utils.metrics import metric
from utils.losses import MaskedLoss, DownstreamLoss
from fvcore.nn import FlopCountAnalysis

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import os
import time

import warnings

warnings.filterwarnings('ignore')


def get_exp_setting(args):
    if args.is_pretrain:
        return '{}_{}_{}_sl{}'.format(
            args.model_id, args.model, args.data, args.seq_len)  # set the name of pretrain model
    return '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_fc{}_eb{}'.format(
        args.model, args.data, args.features, args.seq_len, args.label_len,
        args.pred_len, args.d_model, args.n_heads, args.e_layers,
        args.d_ff, args.factor, args.embed)


class Exp_Main(Exp_Basic):
    def __init__(self, args, setting=None):
        super(Exp_Main, self).__init__(args)
        if setting is None:
            setting = get_exp_setting(args)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        # Writer will output to ./tensorflow/ directory by default
        self.writer = SummaryWriter(path + '/tensorflow/')

    def _calculate_flops(self):
        # Define the input shapes correctly
        input_shape = (1, self.args.seq_len, self.args.enc_in)  # Adjust num_features as needed
        x_mark_enc_shape = (1, self.args.seq_len, self.args.enc_in)  # Example shape, adjust as needed
        x_dec_shape = (1, self.args.label_len + self.args.pred_len, self.args.enc_in)  # Example shape, adjust as needed
        x_mark_dec_shape = (
            1, self.args.label_len + self.args.pred_len, self.args.enc_in)  # Example shape, adjust as needed

        # Create dummy inputs
        dummy_input = torch.randn(input_shape).to(self.device)
        dummy_x_mark_enc = torch.randn(x_mark_enc_shape).to(self.device)
        dummy_x_dec = torch.randn(x_dec_shape).to(self.device)
        dummy_x_mark_dec = torch.randn(x_mark_dec_shape).to(self.device)

        # Perform a forward pass to calculate FLOPS
        flops = FlopCountAnalysis(self.model, (dummy_input, dummy_x_mark_enc, dummy_x_dec, dummy_x_mark_dec))
        print(f"FLOPS: {flops.total()}")
        self.writer.add_scalar("FLOPS", flops.total(), 0)

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'PatchTST': PatchTST,
            # 'TCN': TCN,
            'TCNTorch': TCNTorch,
            'LLMPatchTST': LLMPatchTST
        }
        print(self.args.model)
        selected_model = model_dict[self.args.model]
        if hasattr(selected_model, 'Model'):
            model = selected_model.Model(self.args).float()
        else:
            # If no Model attribute, assume it's already a model class
            model = selected_model(self.args).float()

        # Log model summary to TensorBoard
        model_summary = str(summary(model))
        print(model_summary)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.is_pretrain:
            return MaskedLoss()
        elif self.args.is_finetune:
            return DownstreamLoss()
        else:
            return nn.MSELoss()

    def _process_batch(self, batch, batch_idx=None, data_set=None):
        batch_x, batch_y, batch_x_mark, batch_y_mark = [
            tensor.float().to(self.device) for tensor in batch
        ]

        # Get text descriptions and embeddings if available
        text_descriptions = None
        precomputed_embeddings = None  # text embeddings passed LLM

        if self.args.model == 'LLMPatchTST' and data_set is not None and batch_idx is not None:
            # Get the indices for this batch
            batch_size = batch_x.shape[0]
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data_set.text_descriptions))

            # Get text descriptions
            if hasattr(data_set, 'text_descriptions'):
                text_descriptions = data_set.text_descriptions[start_idx:end_idx]

            # Get pre-computed embeddings if available and enabled
            use_precomputed_embeddings = getattr(self.args, 'use_precomputed_embeddings', False)
            if use_precomputed_embeddings and hasattr(data_set, 'embeddings') and data_set.embeddings is not None:
                # Get embeddings for each sample in the batch
                batch_embeddings = []
                for i in range(start_idx, end_idx):
                    embedding = data_set.get_embedding(i)
                    if embedding is not None:
                        batch_embeddings.append(embedding)

                if batch_embeddings:
                    # Stack embeddings into a batch tensor
                    precomputed_embeddings = torch.stack(batch_embeddings).to(self.device)
                    print(f"Using pre-computed embeddings with shape: {precomputed_embeddings.shape}")

        return batch_x, batch_y, batch_x_mark, batch_y_mark, text_descriptions, precomputed_embeddings

    def _prepare_decoder_input(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return dec_inp

    def _feed_forward_pass(self, batch_x, batch_x_mark, dec_inp, batch_y_mark,
                           text_descriptions, precomputed_embeddings):
        if self.args.model == 'LLMPatchTST':
            outputs = self.model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                text_descriptions=text_descriptions,
                precomputed_embeddings=precomputed_embeddings
            )
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        return outputs

    def _compute_loss(self, batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion, text_descriptions=None,
                      precomputed_embeddings=None):
        outputs = self._feed_forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                          text_descriptions, precomputed_embeddings)

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        return criterion(outputs, batch_y)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, text_descriptions, precomputed_embeddings = self._process_batch(
                    batch, i, vali_data)
                # decoder input
                dec_inp = self._prepare_decoder_input(batch_y)
                # Feed ward pass
                loss = self._compute_loss(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion,
                    text_descriptions, precomputed_embeddings
                )
                total_loss.append(loss.item())  # Convert loss to a Python number
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # Load data
        print("Loading data...")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        print(f"Train data size: {len(train_data)}")
        print(f"Validation data size: {len(vali_data)}")
        print(f"Test data size: {len(test_data)}")


        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # Initialize training steps, early stopping, optimizer, and loss loss
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()  # Set model to training mode
            self.epoch_time = time.time()
            # Log weight distribution at the start of each epoch
            self._log_weight_distribution(epoch)

            for i, batch in enumerate(train_loader):  # enumerate over the training data
                iter_count += 1
                model_optim.zero_grad()

                batch_x, batch_y, batch_x_mark, batch_y_mark, text_descriptions, precomputed_embeddings = self._process_batch(
                    batch, i, train_data)
                # batch_x.shape = [batch_size, seq_len, num_features]
                # batch_y.shape = [batch_size, label_len + pred_len, num_features]
                # text_descriptions.shape = [batch_size, num_descriptions] if available

                # decoder input
                dec_inp = self._prepare_decoder_input(batch_y)

                # encoder - decoder
                loss = self._compute_loss(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, dec_inp, criterion, text_descriptions,
                    precomputed_embeddings
                )
                train_loss.append(loss.item())

                self._log_training_progress(i, epoch, loss, len(train_loader))

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - self.epoch_time))
            train_loss = np.average(train_loss)
            vali_start_time = time.time()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            vali_time = time.time() - vali_start_time

            test_start_time = time.time()
            test_loss = self.vali(test_data, test_loader, criterion)
            test_time = time.time() - test_start_time

            # Log epoch-level metrics
            self._log_epoch_metrics(epoch, train_loss, vali_loss, test_loss, model_optim, vali_time, test_time)

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        # Log the best model checkpoint path and size to TensorBoard
        self.writer.add_text("Model Checkpoint Path", os.path.abspath(best_model_path), 0)
        self.writer.add_scalar("Model Size (MB)", get_model_size(self.model), 0)

        return best_model_path

    def test(self, setting, best_model=None):
        test_data, test_loader = self._get_data(flag='test')
        if best_model is not None:
            print('loading model')
            self.model.load_state_dict(torch.load(best_model))

        preds = []
        trues = []
        # result save
        folder_path = './experiments/results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, text_descriptions, precomputed_embeddings = self._process_batch(
                    batch, i, test_data)

                # decoder input
                dec_inp = self._prepare_decoder_input(batch_y)
                # Feed ward pass
                outputs = self._feed_forward_pass(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                  text_descriptions, precomputed_embeddings)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}\n'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        # save the metrics results into csv file
        import pandas as pd
        metrics_df = pd.DataFrame({
            'mse': [mse],
            'mae': [mae],
            'rmse': [rmse],
            'mape': [mape],
            'mspe': [mspe]
        })
        metrics_df.to_csv(folder_path + 'metrics.csv', index=False)

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        print(f"All the logs saved at {folder_path}")

        # log the metrics to TensorBoard
        self._log_test_metrics(mae, mse, rmse, mape, mspe)

        return

    def _log_training_progress(self, i, epoch, loss, num_iters):
        """
        Log training progress to console and TensorBoard.
        """
        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            self._log_batch_metrics(loss, epoch * num_iters + i)
            # Log weight distribution into TensorBoard
            self._log_weight_distribution(epoch * num_iters + i)

    # log batch-level metrics
    def _log_batch_metrics(self, loss, step):
        self.writer.add_scalar('batch_loss', loss.item(), step)
        self.writer.add_scalar('batch_time', time.time() - self.epoch_time, step)
        return

    def _log_epoch_metrics(self, epoch, train_loss, vali_loss, test_loss, model_optim, vali_time, test_time):
        print(
            f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
        # Log to TensorBoard
        self.writer.add_scalars('loss', {
            'train': train_loss,
            'vali': vali_loss,
            'test': test_loss
        }, epoch)
        self.writer.add_scalar('learning_rate', model_optim.param_groups[0]['lr'], epoch)
        self.writer.add_scalar('vali_time', vali_time, epoch)
        self.writer.add_scalar('test_time', test_time, epoch)

    def _log_test_metrics(self, mae, mse, rmse, mape, mspe):
        # Log test metrics to TensorBoard
        self.writer.add_scalar('test/mae', mae, 0)
        self.writer.add_scalar('test/mse', mse, 0)
        self.writer.add_scalar('test/rmse', rmse, 0)
        self.writer.add_scalar('test/mape', mape, 0)
        self.writer.add_scalar('test/mspe', mspe, 0)

    def _log_weight_distribution(self, step):
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Log only weights, not biases
                self.writer.add_histogram(f'weights/{name}', param.data, step)
                if param.grad is not None:
                    self.writer.add_histogram(f'gradients/{name}', param.grad, step)
