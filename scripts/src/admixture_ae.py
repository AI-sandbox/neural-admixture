import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
import random
import wandb
from plots import generate_plots

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class ZeroOneClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.clamp_(0.,1.)
            w = w/w.sum(axis=1, keepdims=True)

class ConstrainedLinear(torch.nn.Module):
    def __init__ (self, input_size, output_size, hard_init=None, bias=True): 
        super().__init__()
        if hard_init is None:
            log.info('Random decoder initialization.')
            self.W = nn.Parameter(torch.zeros(input_size, output_size)) 
            self.W = nn.init.kaiming_normal_(self.W)
        else:
            log.info('Hardcoded decoder initialization.')
            try:
                assert hard_init.size()[0] == input_size
                assert hard_init.size()[1] == output_size
            except AssertionError as ae:
                ae.args += (f'Decoder initialization tensor does not have the required input size.\n Received: {tuple(hard_init.size())}\n Needed: ({input_size}, {output_size})\n',)
                raise ae
            except Exception as e:
                raise e
            self.W = nn.Parameter(hard_init)
        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.ones(output_size))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        if self.bias:
            return torch.addmm(self.b, x, self.sigmoid(self.W))
        return torch.clamp(torch.mm(x, self.sigmoid(self.W)), 1e-4, 1-1e-4)


class Projector(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

class AdmixtureAE(nn.Module):
    def __init__(self, k=None, num_features=None, encoder_activation=nn.ReLU(),
                 P_init=None, deep_encoder=False, batch_norm=False,
                 lambda_l2=0, dropout=0):
        super().__init__()
        self.k = k
        self.num_features = num_features
        if self.k is not None and self.num_features is not None:
            self.deep_encoder = deep_encoder
            self.encoder_activation = encoder_activation
            self.batch_norm = nn.BatchNorm1d(num_features) if batch_norm else None
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
            self.lambda_l2 = lambda_l2 if lambda_l2 > 1e-6 else 0
            if not self.deep_encoder:
                self.encoder = nn.Linear(self.num_features, self.k, bias=True)            
            elif not batch_norm:
                self.encoder = nn.Sequential(
                        nn.Linear(self.num_features, 512, bias=True),
                        self.encoder_activation,
                        nn.Linear(512, 128, bias=True),
                        self.encoder_activation,
                        nn.Linear(128, 32, bias=True),
                        self.encoder_activation,
                        nn.Linear(32, self.k, bias=True)
                )
            else:
                self.encoder = nn.Sequential(
                        nn.Linear(self.num_features, 512, bias=True),
                        self.encoder_activation,
                        nn.Linear(512, self.k, bias=True)
                )
            self.decoder = ConstrainedLinear(self.k, num_features, hard_init=P_init, bias=False)
            self.sigmoid = nn.Sigmoid()
            self.softmax = nn.Softmax(dim=1)

    def _get_encoder_norm(self):
        if not self.deep_encoder:
            return torch.norm(list(self.encoder.parameters())[0])
        elif self.batch_norm is not None:
            return sum((torch.norm(p) for p in list(self.encoder.parameters())[::3]))
        return sum((torch.norm(p) for p in list(self.encoder.parameters())[::2]))

    def forward(self, X):
        if self.batch_norm is not None:
            X = self.batch_norm(X)
        if self.dropout is not None:
            X = self.dropout(X)
        enc = self.encoder(X)
        del X
        hid_state = self.softmax(enc)
        del enc
        return self.decoder(hid_state), hid_state
        
    def launch_training(self, trX, optimizer, loss_f, num_epochs,
                        device, batch_size=0, valX=None, display_logs=True,
                        loss_weights=None, save_every=10, save_path='../outputs/model.pt',
                        run_name=None, plot_every=0, trY=None, valY=None, seed=42, shuffle=False):
        if plot_every != 0:
            assert trY is not None and valY is not None and valX is not None, 'Labels are needed if plots are to be generated'
        random.seed(seed)
        loss_f_supervised = None
        if self.supervised:
            log.info('Going to train on supervised mode.')
            assert trY is not None and valY is not None, 'Ground truth ancestries needed for supervised mode'
            ancestry_dict = {anc: idx for idx, anc in enumerate(sorted(np.unique(trY)))}
            assert len(ancestry_dict) == self.ks[0], 'Number of ancestries in training ground truth is not equal to the value of k'
            to_idx_mapper = np.vectorize(lambda x: ancestry_dict[x])
            trY_num = to_idx_mapper(trY[:])
            valY_num = to_idx_mapper(valY[:])
            loss_f_supervised = nn.CrossEntropyLoss(reduction='mean')
        for ep in range(num_epochs):
            if optimizer_2 is None or ep % 2 == 0:
                tr_loss, val_loss = self._run_epoch(trX, optimizer, loss_f, batch_size, valX, device, loss_weights, shuffle, loss_f_supervised, trY_num, valY_num)
            else:
                tr_loss, val_loss = self._run_epoch(trX, optimizer_2, loss_f, batch_size, valX, device, loss_weights, shuffle, loss_f_supervised, trY_num, valY_num)
            if run_name is not None and val_loss is not None:
                wandb.log({"tr_loss": tr_loss, "val_loss": val_loss})
            elif run_name is not None:
                wandb.log({"tr_loss": tr_loss})
            assert not math.isnan(tr_loss), 'Training loss is NaN'
            if display_logs:
                log.info(f'[METRICS] EPOCH {ep+1}: mean training loss: {tr_loss}')
                if val_loss is not None:
                    log.info(f'[METRICS] EPOCH {ep+1}: mean validation loss: {val_loss}')
            if save_every*ep > 0 and ep % save_every == 0:
                torch.save(self.state_dict(), save_path)
            if plot_every != 0 and (ep == 0 or (ep+1) % plot_every == 0):
                log.info('Rendering plots for epoch {}'.format(ep+1))
                generate_plots(self, trX, trY, valX, valY, device,
                               batch_size, is_multihead=True,
                               min_k=min(self.ks), max_k=max(self.ks), epoch=ep+1)
        return ep+1

    def _batch_generator(self, X, batch_size=0, shuffle=False, y=None):
        idxs = [i for i in range(X.shape[0])]
        if shuffle:
            random.shuffle(idxs)
        if batch_size < 1:
            yield torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64) if y is not None else None
        else:
            for i in range(0, X.shape[0], batch_size):
                yield torch.tensor(X[sorted(idxs[i:i+batch_size])], dtype=torch.float32), torch.tensor(y[sorted(idxs[i:i+batch_size])], dtype=torch.int64) if y is not None else None

    def _validate(self, valX, loss_f, batch_size, device, loss_f_supervised=None, valY=None):
        acum_val_loss = 0
        with torch.no_grad():
            for X in self._batch_generator(valX, batch_size, y=valY):
                X = X.to(device)
                rec, _ = self(X)
                acum_val_loss += loss_f(rec, X).item()
            if self.lambda_l2 > 1e-6:
                acum_val_loss += self.lambda_l2*self._get_encoder_norm()**2
        return acum_val_loss

        
    def _run_step(self, X, optimizer, loss_f, loss_weights=None,
                  loss_f_supervised=None, trY=None):
        optimizer.zero_grad()
        rec, _ = self(X)
        if loss_weights is not None:
            loss = loss_f(rec, X, loss_weights)
        else:
            loss = loss_f(rec, X)
        if self.lambda_l2 > 1e-6:
            loss += self.lambda_l2*self._get_encoder_norm()**2
        del rec, _
        loss.backward()
        optimizer.step()
        return loss.item()

    def _run_epoch(self, trX, optimizer, loss_f, batch_size, valX,
                   device, loss_weights=None, shuffle=False, loss_f_supervised=None,
                   trY=None, valY=None):
        tr_loss, val_loss = 0, None
        self.train()
        for X, y in self._batch_generator(trX, batch_size, shuffle=shuffle, y=trY if loss_f_supervised is not None else None):
            step_loss = self._run_step(X.to(device), optimizer, loss_f, loss_weights, loss_f_supervised, y.to(device))
            tr_loss += step_loss
        if valX is not None:
            self.eval()
            val_loss = self._validate(valX, loss_f, batch_size, device, loss_f_supervised, valY)
            return tr_loss / trX.shape[0], val_loss / valX.shape[0]
        return tr_loss / trX.shape[0], None
