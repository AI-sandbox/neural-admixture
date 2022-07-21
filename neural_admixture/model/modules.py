import logging
import sys
import torch
import torch.nn as nn

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

class ZeroOneClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            with torch.no_grad():
                w = module.weight.data
                w.clamp_(0.,1.) # 0 <= F_ij <= 1

class NeuralEncoder(nn.Module):
    def __init__(self, input_size, ks):
        super().__init__()
        assert sum([k < 2 for k in ks]) == 0, 'Invalid number of clusters. Requirement: k >= 2'
        self.ks = ks
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, k, bias=True)
                # nn.BatchNorm1d(k)
            ) for k in ks])

    def _get_head_for_k(self, k):
        return self.heads[k-min(self.ks)]

    def forward(self, X):
        outputs = [self._get_head_for_k(k)(X) for k in self.ks]
        return outputs


class NeuralDecoder(nn.Module):
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

