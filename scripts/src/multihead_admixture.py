import logging
import torch
import torch.nn as nn
from admixture_ae import AdmixtureAE, ConstrainedLinear

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class MultiHeadEncoder(nn.Module):
    def __init__(self, input_size, ks):
        super().__init__()
        try:
            assert sum([k < 2 for k in ks]) == 0
        except AssertionError as ae:
            ae.args += ('Invalid number of clusters. Requirement: k >= 2',)
            raise ae
        except Exception as e:
            raise e
        self.ks = ks
        self.heads = nn.ModuleList([nn.Linear(input_size, k, bias=True) for k in ks])
        self.softmax = nn.Softmax(dim=1)

    def _get_head_for_k(self, k):
        return self.heads[k-min(self.ks)]

    def forward(self, X):
        outputs = [self.softmax(self._get_head_for_k(k)(X)) for k in self.ks]
        return outputs


class MultiHeadDecoder(nn.Module):
    def __init__(self, ks, output_size, bias=False, inits=None):
        super().__init__()
        self.ks = ks
        if inits is None:
            self.decoders = nn.ModuleList(
                [ConstrainedLinear(k, output_size, hard_init=inits, bias=bias) for k in self.ks]
            )
        else:
            layers = [None]*len(self.ks)
            for i in range(len(ks)):
                ini = end if i != 0 else 0
                end = ini+self.ks[i]
                layers[i] = ConstrainedLinear(self.ks[i], output_size, hard_init=inits[ini:end], bias=bias)            
            self.decoders = nn.ModuleList(layers)
        assert len(self.decoders) == len(self.ks)

    def _get_decoder_for_k(self, k):
        return self.decoders[k-min(self.ks)]

    def forward(self, hid_states):
        outputs = [self._get_decoder_for_k(self.ks[i])(hid_states[i]) for i in range(len(self.ks))]
        return outputs


class AdmixtureMultiHead(AdmixtureAE):
    def __init__(self, ks, num_features, encoder_activation=nn.ReLU(),
                 P_init=None, batch_norm=False, batch_norm_hidden=False,
                 lambda_l2=0, dropout=0):
        super().__init__(ks[0], num_features)
        self.ks = ks
        self.num_features = num_features
        self.encoder_activation = encoder_activation
        self.batch_norm = nn.BatchNorm1d(num_features) if batch_norm else None
        self.batch_norm_hidden = nn.BatchNorm1d(512) if batch_norm_hidden else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.lambda_l2 = lambda_l2 if lambda_l2 > 1e-6 else 0
        self.common_encoder = nn.Sequential(
                nn.Linear(self.num_features, 512, bias=True),
                self.encoder_activation,
        )
        self.multihead_encoder = MultiHeadEncoder(512, self.ks)
        self.decoders = MultiHeadDecoder(self.ks, num_features, bias=False, inits=P_init)

    def _get_encoder_norm(self):
        return torch.norm(list(self.common_encoder.parameters())[0])

    def forward(self, X):
        if self.batch_norm is not None:
            X = self.batch_norm(X)
        if self.dropout is not None:
            X = self.dropout(X)
        enc = self.common_encoder(X)
        if self.batch_norm_hidden is not None:
            enc = self.batch_norm_hidden(enc)
        hid_states = self.multihead_encoder(enc)
        reconstructions = self.decoders(hid_states)
        return reconstructions, hid_states

    def _run_step(self, X, optimizer, loss_f, loss_weights=None):
        optimizer.zero_grad()
        recs, _ = self(X)
        if loss_weights is not None:
            loss = sum((loss_f(rec, X, loss_weights) for rec in recs))
        else:
            loss = sum((loss_f(rec, X) for rec in recs))
        if self.lambda_l2 > 1e-6:
            loss += self.lambda_l2*self._get_encoder_norm()**2
        loss.backward()
        optimizer.step()
        return loss
    
    def _validate(self, valX, loss_f, batch_size, device):
        acum_val_loss = 0
        with torch.no_grad():
            for X in self._batch_generator(valX, batch_size):
                X = X.to(device)
                recs, _ = self(X)
                acum_val_loss += sum((loss_f(rec, X) for rec in recs)).item()
            if self.lambda_l2 > 1e-6:
                acum_val_loss += self.lambda_l2*self._get_encoder_norm()**2
        return acum_val_loss