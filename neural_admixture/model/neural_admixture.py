import dask
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
from typing import Iterable, Optional, Tuple, Union

from .modules import NeuralDecoder, NeuralEncoder, ZeroOneClipper

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

class NeuralAdmixture(nn.Module):
    """Instantiate Neural ADMIXTURE model

    Args:
        ks (Iterable[int]): different numbers of populations to use. If the list contains more than one value, the multi-head version will run.
        num_features (int): number of SNPs used to train the network.
        encoder_activation (nn.Module, optional): activation function used in the encoder. Defaults to nn.GELU().
        P_init (Optional[torch.Tensor], optional): if provided, corresponds to initialization weights as returned by one of the initialization functions. Defaults to None.
        lambda_l2 (float, optional): L2 regularization strength in the encoder. Smaller values will give quasi-binary predictions. Defaults to 5e-4.
        hidden_size (int, optional): number of neurons in the first linear layer. Defaults to 64.
        freeze_decoder (bool, optional): if set to True, the decoder weights are frozen (useful for reusing other results). Defaults to False.
        supervised (bool, optional): if set to True, will run in supervised mode. Defaults to False.
        supervised_loss_weight (float, optional): weight given to the supervised loss term. Only applied if running in supervised mode. Defaults to 0.05.
    """
    def __init__(self, ks: Iterable[int], num_features: int, encoder_activation: nn.Module=nn.GELU(),
                 P_init: Optional[torch.Tensor]=None, lambda_l2: float=5e-4, hidden_size: int=64,
                 freeze_decoder: bool=False, supervised: bool=False, supervised_loss_weight: float=0.05) -> None:
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

    def forward(self, X: torch.Tensor, only_assignments: bool=False,
                only_hidden_states: bool=False) -> Union[torch.Tensor, Iterable[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the model

        Args:
            X (torch.Tensor): _description_
            only_assignments (bool, optional): if set to True, only return the Q values (used in inference mode for speeding up). Defaults to False.
            only_hidden_states (bool, optional): if set to True, only return logits of the Q values. Defaults to False.

        Returns:
            Union[torch.Tensor, Iterable[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]: SNP reconstruction and Q values (if last two parameters are set to False), otherwise only Q values or logits of Q values.
        """
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

    def launch_training(self, trX: da.core.Array, optimizer: torch.optim.Optimizer,
                        loss_f: torch.nn.modules.loss._Loss, num_epochs: int,
                        device: torch.device, batch_size: int=0, valX: Optional[da.core.Array]=None,
                        save_every: int=10, save_path: str='../outputs/model.pt',
                        trY: Optional[Iterable[str]]=None, valY: Optional[Iterable[str]]=None, seed: int=42,
                        shuffle: bool=True, log_to_wandb: bool=False, tol: float=1e-5, dry_run: bool=False,
                        warmup_epochs: int=10, Q_inits: Optional[Iterable[torch.Tensor]]=None) -> int:
        """Launch training pipeline of the model

        Args:
            trX (da.core.Array): training data matrix
            optimizer (torch.optim.Optimizer): loaded optimizer to use
            loss_f (torch.nn.modules.loss._Loss): instantiated loss function to use
            num_epochs (int): maximum number of epochs. Note that the training might stop earlier if the loss is in a plateau.
            device (torch.device): device to use for training
            batch_size (int, optional): batch size. If 0, will use all the training data in a single batch. Defaults to 0.
            valX (Optional[da.core.Array], optional): _description_. Defaults to None.
            save_every (int, optional): save a checkpoint after this number of epochs. Defaults to 10.
            save_path (str, optional): output path of the trained model. Defaults to '../outputs/model.pt'.
            trY (Optional[Iterable[str]], optional): list of training populations per sample. Defaults to None.
            valY (Optional[Iterable[str]], optional): list of validation populations per sample. Defaults to None.
            seed (int, optional): seed for RNG. Defaults to 42.
            shuffle (bool, optional): whether to shuffle the samples at every epoch. Defaults to True.
            log_to_wandb (bool, optional): whether to log training to wandb. Defaults to False.
            tol (float, optional): tolerance for early stopping. Training will stop if decrease in loss is smaller than this value. Defaults to 1e-5.
            dry_run (bool, optional): whether to run a dry run (no output is written to disk). Defaults to False.
            warmup_epochs (int, optional): number of warmup epochs to bring Q to a good initial solution. If 0, no warmup is performed. Defaults to 10.
            Q_inits(Optional[Iterable[torch.Tensor]], optional): initial Q values for warmup. Defaults to None.
        Returns:
            int: number of actual training epochs ran
        """
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
        log.info("Running warmup epochs...")
        if warmup_epochs > 0:
            if Q_inits is None:
                log.warning("No Q initialization was provided. Skipping warmup epochs.")
            else:
                self.decoders.freeze()
                loss_f_warmup = nn.BCELoss(reduction='mean')
                opt_warmup = torch.optim.AdamW(self.common_encoder.parameters(), lr=1e-5)
                for wep in range(warmup_epochs):
                    _, _ = self._run_warmup_epoch(trX, Q_inits, opt_warmup, loss_f_warmup, batch_size, device, shuffle, epoch_num=wep+1)
                if not self.decoders.force_freeze:
                    self.decoders.unfreeze()
        log.info("Training...")
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
            if not dry_run and save_every*ep > 0 and ep % save_every == 0:
                torch.save(self.state_dict(), save_path)
            if ep > 15 and tol > 0 and self._has_converged(tr_diff, tol):
                log.info(f'Convergence criteria met. Stopping fit after {ep+1} epochs...')
                return ep+1
        log.info(f'Max epochs reached. Stopping fit...')
        return ep+1

    def _get_encoder_norm(self, p: int=2) -> torch.Tensor:
        """Retrieve the sum of the norm of the encoder parameters (for regularization purposes)

        Args:
            p (int, optional): norm to retrieve. Defaults to 2.

        Returns:
            torch.Tensor: sum of norms of the encoder parameters
        """
        shared_params = torch.cat([x.view(-1) for x in self.common_encoder.parameters()])
        multihead_params = torch.cat([x.view(-1) for x in self.multihead_encoder.parameters()])
        return torch.norm(shared_params, p)+torch.norm(multihead_params, p)

    def _run_step(self, X: torch.Tensor, optimizer: torch.optim.Optimizer, loss_f: torch.nn.modules.loss._Loss,
                  loss_f_supervised: Optional[torch.nn.modules.loss._Loss], y: Optional[Iterable[str]],
                  warmup: Optional[bool]=False, Q_inits: Optional[Iterable[torch.Tensor]]=None) -> float:
        """Run a single optimization step
        Args:
            X (torch.Tensor): mini-batch of data
            optimizer (torch.optim.Optimizer): loaded optimizer to use
            loss_f (torch.nn.modules.loss._Loss): instantiated loss function to use
            loss_f_supervised (Optional[torch.nn.modules.loss._Loss]): instantiated supervied loss function to use
            y (Optional[Iterable[str]], optional): list of training populations per sample. Defaults to None.
            warmup (Optional[bool], optional): whether to run a warmup step. Defaults to False.
            Q_inits (Optional[Iterable[torch.Tensor]], optional): list of Q initialization matrices. Defaults to None.

        Returns:
            float: loss value of the mini-batch
        """
        optimizer.zero_grad(set_to_none=True)
        if warmup:
            hid_states = self(X, only_assignments=True)
            recs = None
        else:
            recs, hid_states = self(X)
        if Q_inits is None and not warmup: # Regular step
            loss = sum((loss_f(rec, X) for rec in recs))
        elif Q_inits is not None and warmup: # Warmup step
            loss = sum((loss_f(h, Q_init) for h, Q_init in zip(hid_states, Q_inits)))
        else:
            raise ValueError("Cannot provide Q initialization for a regular step")
        if loss_f_supervised is not None:  # Currently only implemented for single-head architecture!
            mask = y > -1
            loss += sum((self.supervised_loss_weight*loss_f_supervised(h[mask], y[mask]) for h in hid_states))
        if not warmup and self.lambda_l2 > 0:
            loss += self.lambda_l2*self._get_encoder_norm(2)**2
        del recs, hid_states
        loss.backward()
        optimizer.step()
        self.decoders.decoders.apply(self.clipper)
        return loss.item()
    
    def _validate(self, valX: da.core.Array, loss_f: torch.nn.modules.loss._Loss, batch_size: int, device: torch.device,
                  loss_f_supervised: Optional[torch.nn.modules.loss._Loss]=None, y: Optional[Iterable[str]]=None) -> float:
        """_summary_

        Args:
            valX (da.core.Array): validation data matrix
            loss_f (torch.nn.modules.loss._Loss): instantiated loss function to use
            batch_size (int): batch size. Note that the last batch might be smaller.
            device (torch.device): device to use for training
            loss_f_supervied (Optional[torch.nn.modules.loss._Loss]): instantiated supervied loss function to use
            y (Optional[Iterable[str]], optional): list of training populations per sample. Defaults to None.

        Returns:
            float: loss value of the validation set
        """
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

    def batch_generator(self, X, batch_size=0, shuffle=True, y=None, Q_inits=None):
        is_inmem = not isinstance(X, da.core.Array)
        idxs = [i for i in range(X.shape[0])]
        if shuffle:
            random.shuffle(idxs)
        if batch_size < 1:
            batch_size = X.shape[0]
        else:
            for i in range(0, X.shape[0], batch_size):
                with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                    to_yield_X, to_yield_y, to_yield_Y_warmup = X[sorted(idxs[i:i+batch_size])], None, None
                if not is_inmem:
                    to_yield_X = to_yield_X.compute()
                if y is not None:
                    to_yield_y = torch.as_tensor(y[sorted(idxs[i:i+batch_size])], dtype=torch.int64)
                if Q_inits is not None:
                    to_yield_Y_warmup = [Q[sorted(idxs[i:i+batch_size])] for Q in Q_inits]
                yield torch.as_tensor(to_yield_X, dtype=torch.float32), to_yield_y, to_yield_Y_warmup

    def _run_epoch(self, trX, optimizer, loss_f, batch_size, valX,
                   device, shuffle=True, loss_f_supervised=None,
                   trY=None, valY=None, epoch_num=0):
        tr_loss, val_loss = 0, None
        self.train()
        total_b = trX.shape[0]//batch_size+1
        for X, y, _ in tqdm(self.batch_generator(trX, batch_size, shuffle=shuffle, y=trY if loss_f_supervised is not None else None), desc=f"Epoch {epoch_num}", total=total_b):
            step_loss = self._run_step(X.to(device), optimizer, loss_f, loss_f_supervised, y.to(device) if y is not None else None)
            tr_loss += step_loss
        if valX is not None:
            self.eval()
            val_loss = self._validate(valX, loss_f, batch_size, device, loss_f_supervised, valY)
            return tr_loss/trX.shape[0], val_loss/valX.shape[0]
        return tr_loss/trX.shape[0], None

    def _run_warmup_epoch(self, trX, Q_inits, optimizer, loss_f, batch_size,
                          device, shuffle=True, epoch_num=0) -> Tuple[float, Union[float, None]]:
        tr_loss = 0
        total_b = trX.shape[0]//batch_size+1
        self.train()
        for X, _, Ys in tqdm(self.batch_generator(trX, batch_size, shuffle=shuffle, Q_inits=Q_inits), desc=f"Warmup epoch {epoch_num}", total=total_b):
            step_loss = self._run_step(X.to(device), optimizer, loss_f,
                                       None, None, warmup=True,
                                       Q_inits=[Y.to(device) for Y in Ys])
            tr_loss += step_loss
        log.info(f"Warmup training loss: {tr_loss/trX.shape[0]}")# !
        return tr_loss/trX.shape[0], None


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
