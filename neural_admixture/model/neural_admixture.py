import dask.array as da
import json
import logging
import math
import numpy as np
import random
import sys
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from tqdm.auto import tqdm

from .modules import NeuralDecoder, NeuralEncoder, ZeroOneClipper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

class NeuralAdmixture(nn.Module):
    def __init__(self, ks, num_features, encoder_activation=nn.GELU(),
                 P_init=None, lambda_l2=0.0005, hidden_size=64,
                 freeze_decoder=False, supervised=False, supervised_loss_weight=0.05):
        super().__init__()
        self.ks = ks
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.encoder_activation = encoder_activation
        self.supervised = supervised
        self.supervised_loss_weight = supervised_loss_weight
        self.freeze_decoder = freeze_decoder
        self.batch_norm = nn.BatchNorm1d(self.num_features)
        self.lambda_l2 = lambda_l2 if lambda_l2 > 1e-8 else 0
        self.softmax = nn.Softmax(dim=1)
        self.common_encoder = nn.Sequential(
                nn.Linear(self.num_features, self.hidden_size, bias=True),
                self.encoder_activation,
        )
        self.multihead_encoder = NeuralEncoder(self.hidden_size, self.ks)
        self.decoders = NeuralDecoder(self.ks, num_features, bias=False, inits=P_init, freeze=self.freeze_decoder)
        self.clipper = ZeroOneClipper()
        self.decoders.decoders.apply(self.clipper)

    def forward(self, X, only_assignments=False, only_hidden_states=False):
        X = self.batch_norm(X)
        enc = self.common_encoder(X)
        del X
        hid_states = self.multihead_encoder(enc)
        if only_hidden_states:
            return hid_states
        probs = [self.softmax(h) for h in hid_states]
        if only_assignments:
            return probs
        del enc
        return self.decoders(probs), probs

    def launch_training(self, trX, optimizer, loss_f, num_epochs,
                        device, batch_size=0, valX=None,
                        save_every=10, save_path='../outputs/model.pt',
                        trY=None, valY=None, seed=42, shuffle=True, log_to_wandb=False,
                        tol=1e-5):
        random.seed(seed)
        loss_f_supervised, trY_num, valY_num = None, None, None
        tr_losses, val_losses = [], []
        log.info(f'Will stop optimization when difference in objective function between two subsequent iterations is < {tol} or after {num_epochs} epochs.')
        if self.supervised:
            log.info('Going to train on supervised mode.')
            assert trY is not None, 'Training ground truth ancestries needed for supervised mode'
            ancestry_dict = {anc: idx for idx, anc in enumerate(sorted(np.unique([a for a in trY if a  != '-'])))}
            assert len(ancestry_dict) == self.ks[0], 'Number of ancestries in training ground truth is not equal to the value of k'
            ancestry_dict['-'] = -1
            to_idx_mapper = np.vectorize(lambda x: ancestry_dict[x])
            trY_num = to_idx_mapper(trY[:])
            valY_num = to_idx_mapper(valY[:]) if valY is not None else None
            loss_f_supervised = nn.CrossEntropyLoss(reduction='mean')
        log.info("Bringing training data into memory...")
        trX = trX.compute()
        for ep in range(num_epochs):
            tr_loss, val_loss = self._run_epoch(trX, optimizer, loss_f, batch_size, valX, device, shuffle, loss_f_supervised, trY_num, valY_num, epoch_num=ep+1)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            assert not math.isnan(tr_loss), 'Training loss is NaN'
            if log_to_wandb and val_loss is not None:
                wandb.log({"tr_loss": tr_loss, "val_loss": val_loss})
            elif log_to_wandb and val_loss is None:
                wandb.log({"tr_loss": tr_loss})
            tr_diff = tr_losses[-2]-tr_losses[-1] if len(tr_losses) > 1 else 'NaN'
            val_diff = val_losses[-2]-val_losses[-1] if val_loss is not None and len(val_losses) > 1 else 'NaN'
            log.info(f'[METRICS] EPOCH {ep+1}: mean training loss: {tr_loss}, diff: {tr_diff}')
            if val_loss is not None:
                log.info(f'[METRICS] EPOCH {ep+1}: mean validation loss: {val_loss}, diff: {val_diff}')
            if save_every*ep > 0 and ep % save_every == 0:
                torch.save(self.state_dict(), save_path)
            if ep > 0 and tol > 0 and self._has_converged(tr_diff, tol):
                log.info(f'Convergence criteria met. Stopping fit after {ep+1} epochs...')
                return ep+1
        log.info(f'Max epochs reached. Stopping fit...')
        return ep+1

    def _get_encoder_norm(self, p=2):
        shared_params = torch.cat([x.view(-1) for x in self.common_encoder.parameters()])
        multihead_params = torch.cat([x.view(-1) for x in self.multihead_encoder.parameters()])
        return torch.norm(shared_params, p)+torch.norm(multihead_params, p)

    def _run_step(self, X, optimizer, loss_f, loss_f_supervised=None, y=None):
        optimizer.zero_grad(set_to_none=True)
        recs, hid_states = self(X)
        loss = sum((loss_f(rec, X) for rec in recs))
        if loss_f_supervised is not None:  # Currently only implemented for single-head architecture!
            mask = y > -1
            loss += sum((self.supervised_loss_weight*loss_f_supervised(h[mask], y[mask]) for h in hid_states))
        if self.lambda_l2 > 0:
            loss += self.lambda_l2*self._get_encoder_norm(2)**2
        del recs, hid_states
        loss.backward()
        optimizer.step()
        self.decoders.decoders.apply(self.clipper)
        return loss.item()
    
    def _validate(self, valX, loss_f, batch_size, device, loss_f_supervised=None, y=None):
        acum_val_loss = 0
        with torch.inference_mode():
            for X, y_b in self.batch_generator(valX, batch_size, y=y if loss_f_supervised is not None else None):
                X = X.to(device)
                y_b = y_b.to(device) if y_b is not None else None
                recs, hid_states = self(X)
                acum_val_loss += sum((loss_f(rec, X).item() for rec in recs))
                if loss_f_supervised is not None and y_b is not None:
                    mask = y_b > -1
                    acum_val_loss += sum((self.supervised_loss_weight*loss_f_supervised(h[mask], y_b[mask]).item() for h in hid_states))
            if self.lambda_l2 > 1e-6:
                acum_val_loss += self.lambda_l2*self._get_encoder_norm()**2
        return acum_val_loss

    def batch_generator(self, X, batch_size=0, shuffle=True, y=None):
        is_inmem = not isinstance(X, da.core.Array)
        idxs = [i for i in range(X.shape[0])]
        if shuffle:
            random.shuffle(idxs)
        if batch_size < 1:
            yield torch.as_tensor(X.compute() if not is_inmem else X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.int64) if y is not None else None
        else:
            for i in range(0, X.shape[0], batch_size):
                to_yield = X[sorted(idxs[i:i+batch_size])]
                yield torch.as_tensor(to_yield.compute() if not is_inmem else to_yield, dtype=torch.float32), torch.as_tensor(y[sorted(idxs[i:i+batch_size])], dtype=torch.int64) if y is not None else None

    def _run_epoch(self, trX, optimizer, loss_f, batch_size, valX,
                   device, shuffle=True, loss_f_supervised=None,
                   trY=None, valY=None, epoch_num=0):
        tr_loss, val_loss = 0, None
        self.train()
        total_b = trX.shape[0]//batch_size+1
        for X, y in tqdm(self.batch_generator(trX, batch_size, shuffle=shuffle, y=trY if loss_f_supervised is not None else None), desc=f"Epoch {epoch_num}", total=total_b):
            step_loss = self._run_step(X.to(device), optimizer, loss_f, loss_f_supervised, y.to(device) if y is not None else None)
            tr_loss += step_loss
        if valX is not None:
            self.eval()
            val_loss = self._validate(valX, loss_f, batch_size, device, loss_f_supervised, valY)
            return tr_loss / trX.shape[0], val_loss / valX.shape[0]
        return tr_loss / trX.shape[0], None

    @staticmethod
    def _has_converged(diff, tol):
        return diff < tol
        
    @staticmethod
    def _hudsons_fst(pop1, pop2):
        '''
        Computes Hudson's Fst given variant frequencies of two populations.
        '''
        try:
            num = np.mean(((pop1-pop2)**2))
            den = np.mean(np.multiply(pop1, 1-pop2)+np.multiply(pop2, 1-pop1))+1e-7
            return num/den
        except Exception as e:
            log.error(e)
            return np.nan

    def display_divergences(self):
        for i, k in enumerate(self.ks):
            dec = self.decoders.decoders[i].weight.data.cpu().numpy()
            header = '\t'.join([f'Pop{p}' for p in range(k-1)])
            print(f'\nFst divergences between estimated populations: (K = {k})')
            print(f'\t{header}')
            print('Pop0')
            for j in range(1, k):
                print(f'Pop{j}', end='')
                pop2 = dec[:,j]
                for l in range(j):
                    pop1 = dec[:,l]
                    fst = self._hudsons_fst(pop1, pop2)
                    print(f"\t{fst:0.3f}", end="" if l != j-1 else '\n')
        return
    
    def save_config(self, name, save_dir):
        config = {
            'Ks': self.ks,
            'num_snps': self.num_features,
        }
        with open(Path(save_dir)/f"{name}_config.json", 'w') as fb:
            json.dump(config, fb)
        log.info('Configuration file saved.')
        return
