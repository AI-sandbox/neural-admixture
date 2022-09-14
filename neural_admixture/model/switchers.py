import torch.nn as nn
import torch.optim as optim

from . import initializations as init

class Switchers(object):
    """Switcher object for several utilities
    """
    _activations = {
        'relu': lambda x: nn.ReLU(),
        'tanh': lambda x: nn.Tanh(),
        'gelu': lambda x: nn.GELU()
    }

    _initializations = {
        'pckmeans': lambda X, y, k, seed, path, run_name, n_comp, batch_size: init.PCKMeansInitialization.get_decoder_init(X, k, path, run_name, n_comp, seed, batch_size),
        'pcarchetypal': lambda X, y, k, seed, path, run_name, n_comp, batch_size: init.PCArchetypal.get_decoder_init(X, k, path, run_name, n_comp, seed, batch_size),
        'pretrained': lambda X, y, k, seed, path, run_name, n_comp, batch_size: init.PretrainedInitialization.get_decoder_init(X, k, path),
        'supervised': lambda X, y, k, seed, path, run_name, n_comp, batch_size: init.SupervisedInitialization.get_decoder_init(X, y, k)
    }

    _optimizers = {
        'adam': lambda params, lr: optim.Adam(params, lr),
        'sgd': lambda params, lr: optim.SGD(params, lr),
    }

    @classmethod
    def get_switchers(cls):
        return {
            'activations': cls._activations,
            'initializations': cls._initializations,
            'optimizers': cls._optimizers
        }
