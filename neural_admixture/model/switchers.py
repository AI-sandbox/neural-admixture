import torch
from . import initializations as init

class Switchers(object):
    """Switcher object for several utilities
    """
    _activations = {
        'relu': lambda x: torch.nn.ReLU(),
        'tanh': lambda x: torch.nn.Tanh(),
        'gelu': lambda x: torch.nn.GELU()
    }

    _initializations = {
        'kmeans': lambda epochs_P1, epochs_P2, batch_size_P1, batch_size_P2, learning_rate_P1_P,
                    learning_rate_f2, K, seed, init_path, name, n_components, data, device, 
                    num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight: 
            
            init.KMeansInitialization.get_decoder_init(epochs_P1, epochs_P2, batch_size_P1, batch_size_P2, learning_rate_P1_P,
                                                learning_rate_f2, K, seed, init_path, name, n_components, data, device, 
                                                num_gpus, hidden_size, activation, master, num_cpus),

        'gmm': lambda epochs_P1, epochs_P2, batch_size_P1, batch_size_P2, learning_rate_P1_P,
                    learning_rate_f2, K, seed, init_path, name, n_components, data, device, 
                    num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight: 
            
            init.GMMInitialization.get_decoder_init(epochs_P1, epochs_P2, batch_size_P1, batch_size_P2, learning_rate_P1_P,
                                                learning_rate_f2, K, seed, init_path, name, n_components, data, device, 
                                                num_gpus, hidden_size, activation, master, num_cpus),
        
        'supervised': lambda epochs_P1, epochs_P2, batch_size_P1, batch_size_P2, learning_rate_P1_P,
                    learning_rate_f2, K, seed, init_path, name, n_components, data, device, 
                    num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight: 
            
            init.SupervisedInitialization.get_decoder_init(epochs_P1, epochs_P2, batch_size_P1, batch_size_P2, learning_rate_P1_P,
                                                learning_rate_f2, K, seed, init_path, name, n_components, data, device, 
                                                num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight),
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
