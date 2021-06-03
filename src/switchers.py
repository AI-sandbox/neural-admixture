import sys
sys.path.append('..')
import model.initializations as init
import torch.nn as nn
import torch.optim as optim

class Switchers(object):
    _activations = {
        'relu': lambda x: nn.ReLU(),
        'tanh': lambda x: nn.Tanh()
    }


    _data = {
        'CHM-1': lambda path: (f'{path}/CHM-1/train.h5', f'{path}/CHM-1/validation.h5'),
        'CHM-22': lambda path: (f'{path}/CHM-22/train.h5', f'{path}/CHM-22/validation.h5'),
        'CHM-22-SIM': lambda path: (f'{path}/CHM-22-SIM/train.h5', f'{path}/CHM-22-SIM/validation.h5')
    }

    _initializations = {
        'random': lambda X, y, k, batch_size, seed, path: None,
        'mean_SNPs': lambda X, y, k, batch_size, seed, path: init.SNPsMeanInitialization.get_decoder_init(X, k),
        'mean_random': lambda X, y, k, batch_size, seed, path: init.RandomMeanInitialization.get_decoder_init(X, k),
        'kmeans': lambda X, y, k, batch_size, seed, path: init.KMeansInitialization.get_decoder_init(X, k, False, False, batch_size, seed),
        'minibatch_kmeans': lambda X, y, k, batch_size, seed, path: init.KMeansInitialization.get_decoder_init(X, k, True, False, batch_size, seed),
        'kmeans++': lambda X, y, k, batch_size, seed, path: init.KMeansPlusPlusInitialization.get_decoder_init(X, k, seed),
        'binomial': lambda X, y, k, batch_size, seed, path: init.BinomialInitialization.get_decoder_init(X, k),
        'pca': lambda X, y, k, batch_size, seed, path: init.PCAInitialization.get_decoder_init(X, k),
        'admixture': lambda X, y, k, batch_size, seed, path: init.AdmixtureInitialization.get_decoder_init(X, k, path),
        'pckmeans': lambda X, y, k, batch_size, seed, path: init.PCKMeansInitialization.get_decoder_init(X, k, path),
        'supervised': lambda X, y, k, batch_size, seed, path: init.SupervisedInitialization.get_decoder_init(X, y, k)
    }

    _optimizers = {
        'adam': lambda params, lr: optim.Adam(params, lr),
        'sgd': lambda params, lr: optim.SGD(params, lr)
    }

    @classmethod
    def get_switchers(cls):
        return {
            'activations': cls._activations,
            'data': cls._data,
            'initializations': cls._initializations,
            'optimizers': cls._optimizers
        }
