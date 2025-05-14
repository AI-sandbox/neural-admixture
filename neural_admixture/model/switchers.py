import torch
from . import initializations as init

class Switchers(object):
    """Switcher object for several utilities
    """
    _activations = {
        'relu': lambda x: torch.nn.ReLU(inplace=True),
        'tanh': lambda x: torch.nn.Tanh(),
        'gelu': lambda x: torch.nn.GELU()
    }

    _initializations = {
        'random': lambda epochs, batch_size, learning_rate, K, seed, n_components, data, device, 
                    num_gpus, hidden_size, activation, master, V, num_cpus, has_missing: 
            
            init.RandomInitialization.get_decoder_init(epochs, batch_size, learning_rate, K, seed, n_components, data, device, 
                                                num_gpus, hidden_size, activation, master, V, num_cpus, has_missing),
        }

    @classmethod
    def get_switchers(cls) -> dict[str, object]:
        """
        Returns:
        - dict[str, object]: A dictionary where the keys are strings ('activations', 'initializations'),
          and the values are the corresponding class-level attributes.
        """
        return {
            'activations': cls._activations,
            'initializations': cls._initializations,
        }
