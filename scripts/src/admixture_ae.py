import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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

    def forward(self, x):
        if self.bias:
            return torch.addmm(self.b, x, torch.sigmoid(self.W))
        return torch.mm(x, torch.sigmoid(self.W)) 


class Projector(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass

class AdmixtureAE(nn.Module):
    def __init__(self, k, num_features, encoder_activation=nn.ReLU(),
                 P_init=None, deep_encoder=False, batch_norm=False,
                 lambda_l2=0, dropout=0):
        super().__init__()
        self.k = k
        self.num_features = num_features
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
        hid_state = self.softmax(enc)
        reconstruction = self.decoder(hid_state)
        return reconstruction, hid_state
        
    def launch_training(self, trX, optimizer, loss_f, num_epochs, device, batch_size=0, valX=None, display_logs=True, loss_weights=None, save_every=10, save_path='../outputs/model.pt', run_name=None):
        for ep in range(num_epochs):
            tr_loss, val_loss = self._run_epoch(trX, optimizer, loss_f, batch_size, valX, device, loss_weights)
            if run_name is not None and val_loss is not None:
                wandb.log({"tr_loss": tr_loss, "val_loss": val_loss})
            elif run_name is not None:
                wandb.log({"tr_loss": tr_loss})
            try:
                assert not math.isnan(tr_loss)
            except AssertionError as ae:
                ae.args += ('Training loss is NaN',)
                raise ae
            except Exception as e:
                raise e
            if display_logs:
                log.info(f'[METRICS] EPOCH {ep+1}: mean training loss: {tr_loss}')
                if val_loss is not None:
                    log.info(f'[METRICS] EPOCH {ep+1}: mean validation loss: {val_loss}')
            if save_every*ep > 0 and ep % save_every == 0:
                torch.save(self.state_dict(), save_path)
        return ep+1

    def _batch_generator(self, X, batch_size=0):
        if batch_size < 1:
            yield torch.tensor(X, dtype=torch.float32)
        else:
            for i in range(0, X.shape[0], batch_size):
                yield torch.tensor(X[i:i+batch_size], dtype=torch.float32)

    def _validate(self, valX, loss_f, batch_size, device):
        acum_val_loss = 0
        with torch.no_grad():
            for X in self._batch_generator(valX, batch_size):
                X = X.to(device)
                rec, _ = self(X)
                acum_val_loss += loss_f(rec, X).item()
            if self.lambda_l2 > 1e-6:
                acum_val_loss += self.lambda_l2*self._get_encoder_norm()**2
        return acum_val_loss

        
    def _run_step(self, X, optimizer, loss_f, loss_weights=None):
        optimizer.zero_grad()
        rec, _ = self(X)
        if loss_weights is not None:
            loss = loss_f(rec, X, loss_weights)
        else:
            loss = loss_f(rec, X)
        if self.lambda_l2 > 1e-6:
            loss += self.lambda_l2*self._get_encoder_norm()**2
        loss.backward()
        optimizer.step()
        return loss

    def _run_epoch(self, trX, optimizer, loss_f, batch_size, valX, device, loss_weights=None):
        tr_loss, val_loss = 0, None
        self.train()
        for X in self._batch_generator(trX, batch_size):
            step_loss = self._run_step(X.to(device), optimizer, loss_f, loss_weights)
            tr_loss += step_loss.item()
        if valX is not None:
            self.eval()
            val_loss = self._validate(valX, loss_f, batch_size, device)
            return tr_loss / trX.shape[0], val_loss / valX.shape[0]
        return tr_loss / trX.shape[0], None
