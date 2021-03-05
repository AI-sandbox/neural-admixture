import custom_losses as cl
import torch.nn as nn
import torch.optim as optim
from initializations import KMeansInitialization, RandomMeanInitialization

class Switchers(object):
    _activations = {
        'relu': lambda device, frac: nn.ReLU(),
        'tanh': lambda device, frac: nn.Tanh()
    }
    _losses = {
        'mse': lambda device, frac: nn.MSELoss(reduction='sum'),
        'bce': lambda device, frac: nn.BCELoss(reduction='sum'),
        'wbce': lambda device, frac: WeightedBCE(),
        'bce_mask': lambda device, frac: MaskedBCE(device, mask_frac=frac),
        'mse_mask': lambda device, frac: MaskedMSE(device, mask_frac=frac) 
    }
    _initializations = {
        'random': lambda X, k, batch_size, seed: None,
        'mean_random': lambda X, k, batch_size, seed: RandomMeanInitialization.get_decoder_init(X),
        'kmeans': lambda X, k, batch_size, seed: KMeansInitialization.get_decoder_init(X, k, False, False, batch_size, seed),
        'minibatch_kmeans': lambda X, k, batch_size, seed: KMeansInitialization.get_decoder_init(X, k, True, False, batch_size, seed),
        'kmeans_logit': lambda X, k, batch_size, seed: KMeansInitialization.get_decoder_init(X, k, False, True, batch_size, seed),
        'minibatch_kmeans_logit': lambda X, k, batch_size, seed: KMeansInitialization.get_decoder_init(X, k, True, True, batch_size, seed)
    }
    _optimizers = {
        'adam': lambda params, lr: optim.Adam(params, lr),
        'sgd': lambda params, lr: optim.SGD(params, lr)
    }

    @classmethod
    def get_switchers(cls):
        return {
            'losses': cls._losses,
            'activations': cls._activations,
            'initializations': cls._initializations,
            'optimizers': cls._optimizers
        }
