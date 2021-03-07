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


class AdmixtureMultiHead(AdmixtureAE):
    def __init__(self, ks, num_features, encoder_activation=nn.ReLU(),
                 P_init=None, batch_norm=False,
                 lambda_l2=0, dropout=0):
        super().__init__(ks[0], num_features)
        self.ks = ks
        self.num_features = num_features
        self.encoder_activation = encoder_activation
        self.batch_norm = nn.BatchNorm1d(num_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.lambda_l2 = lambda_l2 if lambda_l2 > 1e-6 else 0
        self.common_encoder = nn.Sequential(
                nn.Linear(self.num_features, 512, bias=True),
                self.encoder_activation,
        )
        self.multihead_encoder = MultiHeadEncoder(512, ks)
        self.decoder = ConstrainedLinear(sum(self.ks), num_features, hard_init=P_init, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def _get_encoder_norm(self):
        return sum((torch.norm(p) for p in list(self.encoder.parameters())[::2]))

    def forward(self, X):
        if self.batch_norm is not None:
            X = self.batch_norm(X)
        if self.dropout is not None:
            X = self.dropout(X)
        enc = self.common_encoder(X)
        hid_states = self.multihead_encoder(enc)
        merged_states = self.softmax(torch.cat(hid_states, dim=1))
        reconstruction = self.decoder(merged_states)
        return reconstruction, hid_states