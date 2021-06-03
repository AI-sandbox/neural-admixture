import logging
import math
import numpy as np
import random
import torch
import torch.nn as nn
# import wandb
from .modules import NeuralDecoder, NeuralEncoder, ZeroOneClipper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class NeuralAdmixture(nn.Module):
    def __init__(self, ks, num_features, encoder_activation=nn.ReLU(),
                 P_init=None, lambda_l2=0, linear=True, hidden_size=512,
                 freeze_decoder=False, supervised=False):
        super().__init__()
        self.ks = ks
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.encoder_activation = encoder_activation
        self.linear = linear
        self.supervised = supervised
        self.freeze_decoder = freeze_decoder
        self.batch_norm = nn.BatchNorm1d(self.num_features)
        self.lambda_l2 = lambda_l2 if lambda_l2 > 1e-6 else 0
        self.softmax = nn.Softmax(dim=1)
        self.common_encoder = nn.Sequential(
                nn.Linear(self.num_features, self.hidden_size, bias=True),
                self.encoder_activation,
        )
        self.multihead_encoder = NeuralEncoder(self.hidden_size, self.ks)
        if self.linear:
            self.decoders = NeuralDecoder(self.ks, num_features, bias=False, inits=P_init, freeze=self.freeze_decoder)
            self.clipper = ZeroOneClipper()
            self.decoders.decoders.apply(self.clipper)
        else:
            self.decoders = NonLinearMultiHeadDecoder(self.ks, num_features, bias=True,
                                                      hidden_size=self.hidden_size,
                                                      hidden_activation=self.encoder_activation)

    def forward(self, X, only_assignments=False):
        X = self.batch_norm(X)
        enc = self.common_encoder(X)
        del X
        hid_states = self.multihead_encoder(enc)
        probs = [self.softmax(h) for h in hid_states]
        if only_assignments:
            return probs
        del enc
        return self.decoders(probs), hid_states

    def launch_training(self, trX, optimizer, loss_f, num_epochs,
                        device, batch_size=0, valX=None, display_logs=True,
                        save_every=10, save_path='../outputs/model.pt',
                        run_name=None, plot_every=0, trY=None, valY=None, seed=42, shuffle=False):
        if plot_every != 0:
            assert trY is not None and valY is not None and valX is not None, 'Labels are needed if plots are to be generated'
        random.seed(seed)
        loss_f_supervised, trY_num, valY_num = None, None, None
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
            tr_loss, val_loss = self._run_epoch(trX, optimizer, loss_f, batch_size, valX, device, shuffle, loss_f_supervised, trY_num, valY_num)
            # if run_name is not None and val_loss is not None:
            #     wandb.log({"tr_loss": tr_loss, "val_loss": val_loss})
            # elif run_name is not None:
            #     wandb.log({"tr_loss": tr_loss})
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

    def _get_encoder_norm(self):
        return torch.norm(list(self.common_encoder.parameters())[0])

    def _run_step(self, X, optimizer, loss_f, loss_f_supervised=None, y=None):
        optimizer.zero_grad()
        recs, hid_states = self(X)
        loss = sum((loss_f(rec, X) for rec in recs)) if self.linear else loss_f(recs, X)
        if loss_f_supervised is not None:  # Currently only implemented for single-head architecture!
            loss += sum((loss_f_supervised(h, y) for h in hid_states))
        if self.lambda_l2 > 1e-6:
            loss += self.lambda_l2*self._get_encoder_norm()**2
        del recs, hid_states
        loss.backward()
        optimizer.step()
        if self.linear:
            self.decoders.decoders.apply(self.clipper)
        return loss.item()
    
    def _validate(self, valX, loss_f, batch_size, device, loss_f_supervised=None, y=None):
        acum_val_loss = 0
        with torch.no_grad():
            for X, y_b in self._batch_generator(valX, batch_size, y=y if loss_f_supervised is not None else None):
                X = X.to(device)
                y_b = y_b.to(device) if y_b is not None else None
                recs, hid_states = self(X)
                acum_val_loss += sum((loss_f(rec, X).item() for rec in recs)) if self.linear else loss_f(recs, X).item()
                if loss_f_supervised is not None:
                    acum_val_loss += sum((loss_f_supervised(h, y_b).item() for h in hid_states))
            if self.lambda_l2 > 1e-6:
                acum_val_loss += self.lambda_l2*self._get_encoder_norm()**2
        return acum_val_loss

    def _batch_generator(self, X, batch_size=0, shuffle=False, y=None):
        idxs = [i for i in range(X.shape[0])]
        if shuffle:
            random.shuffle(idxs)
        if batch_size < 1:
            yield torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64) if y is not None else None
        else:
            for i in range(0, X.shape[0], batch_size):
                yield torch.tensor(X[sorted(idxs[i:i+batch_size])], dtype=torch.float32), torch.tensor(y[sorted(idxs[i:i+batch_size])], dtype=torch.int64) if y is not None else None

    def _run_epoch(self, trX, optimizer, loss_f, batch_size, valX,
                   device, shuffle=False, loss_f_supervised=None,
                   trY=None, valY=None):
        tr_loss, val_loss = 0, None
        self.train()
        for X, y in self._batch_generator(trX, batch_size, shuffle=shuffle, y=trY if loss_f_supervised is not None else None):
            step_loss = self._run_step(X.to(device), optimizer, loss_f, loss_f_supervised, y.to(device) if y is not None else None)
            tr_loss += step_loss
        if valX is not None:
            self.eval()
            val_loss = self._validate(valX, loss_f, batch_size, device, loss_f_supervised, valY)
            return tr_loss / trX.shape[0], val_loss / valX.shape[0]
        return tr_loss / trX.shape[0], None
