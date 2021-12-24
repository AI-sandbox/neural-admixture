import sys
import model.initializations as init
import torch.nn as nn
import torch.optim as optim

class Switchers(object):
    _activations = {
        'relu': lambda x: nn.ReLU(),
        'tanh': lambda x: nn.Tanh(),
        'gelu': lambda x: nn.GELU()
    }

    _initializations = {
        'pckmeans': lambda X, y, k, seed, path, run_name, n_comp: init.PCKMeansInitialization.get_decoder_init(X, k, path, run_name, n_comp),
        'pcarchetypal': lambda X, y, k, seed, path, run_name, n_comp: init.PCArchetypal.get_decoder_init(X, k, path, run_name, n_comp, seed),
        'pretrained': lambda X, y, k, seed, path, run_name, n_comp: init.PretrainedInitialization.get_decoder_init(X, k, path),
        'supervised': lambda X, y, k, seed, path, run_name, n_comp: init.SupervisedInitialization.get_decoder_init(X, y, k)

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
