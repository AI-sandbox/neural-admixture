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
        'kmeans': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
                    num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight: 
            
            init.KMeansInitialization.get_decoder_init(epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
                                                num_gpus, hidden_size, activation, master, num_cpus),

        'gmm': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
                    num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight: 
            
            init.GMMInitialization.get_decoder_init(epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
                                                num_gpus, hidden_size, activation, master, num_cpus),
        
        'random': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
                    num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight: 
            
            init.RandomInitialization.get_decoder_init(epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
                                                num_gpus, hidden_size, activation, master, num_cpus),
        
        'supervised': lambda epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
                    num_gpus, hidden_size, activation, master, num_cpus, y, supervised_loss_weight: 
            
            init.SupervisedInitialization.get_decoder_init(epochs, batch_size, learning_rate, K, seed, init_path, name, n_components, data, q_nrm, device, 
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
