import logging
import torch
import torch.nn as nn
from admixture_ae import AdmixtureAE, ZeroOneClipper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class MultiHeadEncoder(nn.Module):
    def __init__(self, input_size, ks):
        super().__init__()
        assert sum([k < 2 for k in ks]) == 0, 'Invalid number of clusters. Requirement: k >= 2'
        self.ks = ks
        self.heads = nn.ModuleList([nn.Linear(input_size, k, bias=True) for k in ks])

    def _get_head_for_k(self, k):
        return self.heads[k-min(self.ks)]

    def forward(self, X):
        outputs = [self._get_head_for_k(k)(X) for k in self.ks]
        return outputs


class MultiHeadDecoder(nn.Module):
    def __init__(self, ks, output_size, bias=False, inits=None, freeze=False):
        super().__init__()
        self.ks = ks
        self.freeze = freeze
        if inits is None:
            self.decoders = nn.ModuleList(
                [nn.Linear(k, output_size, bias=bias) for k in self.ks]
            )
            if self.freeze:
                log.warn('Not going to freeze weights as no initialization was provided.')   
        else:
            layers = [None]*len(self.ks)
            for i in range(len(ks)):
                ini = end if i != 0 else 0
                end = ini+self.ks[i]
                layers[i] = nn.Linear(self.ks[i], output_size, bias=bias)
                layers[i].weight = torch.nn.Parameter(inits[ini:end].clone().detach().requires_grad_(not self.freeze).T)
            self.decoders = nn.ModuleList(layers)
            if self.freeze:
                log.info('Decoder weights will be frozen.')
        assert len(self.decoders) == len(self.ks)

    def _get_decoder_for_k(self, k):
        return self.decoders[k-min(self.ks)]

    def forward(self, hid_states):
        outputs = [torch.clamp(self._get_decoder_for_k(self.ks[i])(hid_states[i]), 0, 1) for i in range(len(self.ks))]
        return outputs

class NonLinearMultiHeadDecoder(nn.Module):
    def __init__(self, ks, output_size, bias=False,
                 hidden_size=512, hidden_activation=nn.ReLU(),
                 inits=None):
        super().__init__()
        self.ks = ks
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.heads_decoder = nn.Linear(sum(self.ks), self.hidden_size, bias=bias)
        self.common_decoder = nn.Linear(self.hidden_size, self.output_size)
        self.nonlinearity = hidden_activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, hid_states):
        if len(hid_states) > 1:
            concat_states = torch.cat(hid_states, 1)
        else:
            concat_states = hid_states[0]
        dec = self.nonlinearity(self.heads_decoder(concat_states))
        rec = self.sigmoid(self.common_decoder(dec))
        return rec

class AdmixtureMultiHead(AdmixtureAE):
    def __init__(self, ks, num_features, encoder_activation=nn.ReLU(),
                 P_init=None, batch_norm=False, batch_norm_hidden=False,
                 lambda_l2=0, dropout=0, pooling=1, linear=True,
                 hidden_size=512, freeze_decoder=False, supervised=False):
        super().__init__()
        self.ks = ks
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.encoder_activation = encoder_activation
        self.pooling_factor = pooling
        self.linear=linear
        self.supervised = supervised
        self.freeze_decoder = freeze_decoder
        self.encoder_input_size = num_features//self.pooling_factor+1 if self.pooling_factor > 1 else num_features
        self.batch_norm = nn.BatchNorm1d(self.encoder_input_size) if batch_norm else None
        self.batch_norm_hidden = nn.BatchNorm1d(512) if batch_norm_hidden else None
        self.pool = nn.AvgPool1d(self.pooling_factor, padding=self.pooling_factor//2) if self.pooling_factor > 1 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.lambda_l2 = lambda_l2 if lambda_l2 > 1e-6 else 0
        self.softmax = nn.Softmax(dim=1)
        self.common_encoder = nn.Sequential(
                nn.Linear(self.encoder_input_size, self.hidden_size, bias=True),
                self.encoder_activation,
        )
        self.multihead_encoder = MultiHeadEncoder(self.hidden_size, self.ks)
        if self.linear:
            self.decoders = MultiHeadDecoder(self.ks, num_features, bias=False, inits=P_init, freeze=self.freeze_decoder)
            self.clipper = ZeroOneClipper()
            self.decoders.decoders.apply(self.clipper)

        else:
            self.decoders = NonLinearMultiHeadDecoder(self.ks, num_features, bias=True,
                                                      hidden_size=self.hidden_size,
                                                      hidden_activation=self.encoder_activation)

    def _get_encoder_norm(self):
        return torch.norm(list(self.common_encoder.parameters())[0])

    def forward(self, X, only_assignments=False):
        if self.pool is not None:
            X = self.pool(X.view(-1,1,self.num_features)).view(-1,self.encoder_input_size)
        if self.batch_norm is not None:
            X = self.batch_norm(X)
        if self.dropout is not None:
            X = self.dropout(X)
        enc = self.common_encoder(X)
        del X
        if self.batch_norm_hidden is not None:
            enc = self.batch_norm_hidden(enc)
        hid_states = self.multihead_encoder(enc)
        probs = [self.softmax(h) for h in hid_states]
        if only_assignments:
            return probs
        del enc
        return self.decoders(probs), hid_states

    def _run_step(self, X, optimizer, loss_f, loss_weights=None, loss_f_supervised=None, y=None):
        optimizer.zero_grad()
        recs, hid_states = self(X)
        if loss_weights is not None:
            loss = sum((loss_f(rec, X, loss_weights) for rec in recs)) if self.linear else loss_f(recs, X, loss_weights)
        else:
            loss = sum((loss_f(rec, X) for rec in recs)) if self.linear else loss_f(recs, X)
        if loss_f_supervised is not None:  # Should only be used for a single-head architecture!
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
                y_b = y_b.to(device) if loss_f_supervised is not None else None
                recs, hid_states = self(X)
                acum_val_loss += sum((loss_f(rec, X).item() for rec in recs)) if self.linear else loss_f(recs, X).item()
                if loss_f_supervised is not None:
                    acum_val_loss += sum((loss_f_supervised(h, y_b).item() for h in hid_states))
            if self.lambda_l2 > 1e-6:
                acum_val_loss += self.lambda_l2*self._get_encoder_norm()**2
        return acum_val_loss
