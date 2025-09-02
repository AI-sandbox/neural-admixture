import logging
import sys
import json
import torch

from pathlib import Path
from typing import Optional, Tuple, List
from tqdm.auto import tqdm

from ..src.loaders import dataloader_admixture

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class NeuralEncoder(torch.nn.Module):
    """
    Neural network encoder component. Creates a separate linear head
    for each specified value of k in the provided ks list.
    """
    def __init__(self, input_size, ks):
        """
        Args:
            input_size (int): Dimension of the input features (output of common encoder).
            ks (list[int]): A list of K values for which to create encoder heads.
        """
        super().__init__()
        self.ks = sorted(ks)
        self.min_k_val = min(self.ks)
        self.heads = torch.nn.ModuleList([torch.nn.Linear(input_size, k_val, bias=True) for k_val in self.ks])

    def _get_head_for_k(self, k_val):
        """Retrieves the specific encoder head for a given K value."""
        index = k_val - self.min_k_val
        if index < 0 or index >= len(self.heads):
            raise ValueError(f"K value {k_val} not found in the specified ks list {self.ks}")
        return self.heads[index]

    def forward(self, X):
        """
        Forward pass through the encoder.

        Args:
            X (torch.Tensor): Input data tensor from the common encoder.

        Returns:
            list[torch.Tensor]: A list of output tensors (hidden states), one for each K in self.ks.
        """
        outputs = [self._get_head_for_k(k_val)(X) for k_val in self.ks]
        return outputs

class NeuralDecoder(torch.nn.Module):
    """
    Neural network decoder component. Creates a separate linear decoder
    for each specified value of k, initialized with corresponding parts
    of the initial P matrix.
    """
    def __init__(self, output_size, inits, ks):
        """
        Args:
            output_size (int): Dimension of the output (number of markers M).
            inits (torch.Tensor): The initial P matrix (M, sum(ks)).        
            ks (list[int]): A list of K values for which to create decoders.
        """
        super().__init__()
        self.output_size = output_size
        self.ks = sorted(ks) 
        self.min_k_val = min(self.ks)
    
        layers = [None]*len(self.ks)
        ini = 0
        for i in range(len(self.ks)):
            end = ini+self.ks[i]
            layers[i] = torch.nn.Linear(self.ks[i], output_size, bias=False)
            layers[i].weight = torch.nn.Parameter(inits[ini:end].T)
            ini = end
        self.decoders = torch.nn.ModuleList(layers)
        
    def _get_decoder_for_k(self, k_val):
        """Retrieves the specific decoder for a given K value."""
        index = k_val - self.min_k_val
        return self.decoders[index]

    def forward(self, probs):
        """
        Forward pass through the decoder.

        Args:
            probs (list[torch.Tensor]): A list of probability tensors.

        Returns:
            list[torch.Tensor]: A list of output tensors (reconstructions).
        """
        outputs = []
        for i, k_val in enumerate(self.ks):
            decoder = self._get_decoder_for_k(self.ks[i])
            output = decoder(probs[i])
            outputs.append(torch.clamp_(output, 0, 1))
        return outputs

class Q_P(torch.nn.Module):
    """
    Q_P model.

    Args:
        hidden_size (int): The size of the hidden layer.
        num_features (int): The number of features in the input data.
        k (int): The number of output classes or components.
        V (torch.Tensor): The projection matrix used to map inputs to PCA space.
        P (Optional[torch.Tensor], optional): The P matrix to be optimized. Defaults to None.
        is_train (bool): Indicates whether the model is in training mode (True) or inference mode (False). Defaults to True.
    """
    def __init__(self, hidden_size: int, num_features: int, V: torch.Tensor=None, P: torch.Tensor=None,
                ks_list: List=[], is_train: bool=True) -> None:
        """
        Initialize the Q_P module with the given parameters.

        Args:
            hidden_size (int): The size of the hidden layer.
            num_features (int): The number of features in the input data.
            k (int): The number of output classes or components.
            activation (torch.nn.Module): The activation function to use in the encoder.
            V (torch.Tensor): The projection matrix used to map inputs to PCA space.
            P (Optional[torch.Tensor], optional): The P matrix to be optimized. Defaults to None.
            is_train (bool): Indicates whether the model is in training mode (True) or inference mode (False). Defaults to True.
        """
        super(Q_P, self).__init__()
        self.ks_list = ks_list
        
        if V is not None:
            self.V = torch.nn.Parameter(V)
        else:
            self.V = None
        
        self.num_features = num_features
        self.batch_norm = torch.nn.RMSNorm(self.num_features, eps=1e-8)
        self.encoder_activation = torch.nn.ReLU(inplace=True)
        self.hidden_size = hidden_size
        self.common_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, self.hidden_size, bias=True),
            self.encoder_activation)
        self.multihead_encoder = NeuralEncoder(self.hidden_size, ks=self.ks_list)
        if P is not None:
            self.decoders = NeuralDecoder(self.num_features, P, ks=self.ks_list)
        self.softmax = torch.nn.Softmax(dim=1)
        
        if is_train:
            self.return_func = self._return_training
        else:
            self.return_func = self._return_infer
                
    def _return_training(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoders(probs), probs
    
    def _return_infer(self, probs: torch.Tensor) -> torch.Tensor:
        return probs
     
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass with the given batch of input data.

        Args:
            X (torch.Tensor): A tensor of input data.

        Returns:
            Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]: 
                - If training: A tuple containing the transformed tensor (clamped between 0 and 1) and the probability tensor.
                - If inference: The probability tensor.
        """
        X = X.float() / 2
        X = torch.where(X == 1.5, 0.0, X)
        
        X_pca = X@self.V
        X_pca = self.batch_norm(X_pca)
        enc = self.common_encoder(X_pca)
        hid_states = self.multihead_encoder(enc)
        probs = [self.softmax(h) for h in hid_states]
        return self.return_func(probs), X

    @torch.no_grad()
    def restrict_P(self):
        """
        Restrict the values of P matrix within the range [0, 1].
        """
        for dec in self.decoders.decoders:
            dec.weight.data.clamp_(0., 1.)
    
    def create_custom_adam(self, device: torch.device, lr: float=1e-5) -> torch.optim.Adam:
        """
        Creates a custom Adam optimizer with different learning rates for different phases.

        Args:
            lr (float): Learning rate for all parameters.

        Returns:
            optim.Adam: The Adam optimizer configured with the specified learning rates.
        """
        p = [
        {'params': self.multihead_encoder.parameters(), 'lr': lr},
        {'params': self.common_encoder.parameters(), 'lr': lr},
        {'params': self.batch_norm.parameters(), 'lr': lr},
        {'params': self.V, 'lr': lr},
        {'params': self.decoders.parameters(), 'lr': lr}
        ]
        return torch.optim.Adam(p, betas=[0.9, 0.95], fused=device.type != 'mps')
    
    def save_config(self, name: str, save_dir: str) -> None:
        """
        Saves the model configuration to a JSON file in the specified directory.

        Args:
            name (str): The name of the configuration file (without extension).
            save_dir (str): The directory where the configuration file should be saved.
        """
        _activations = {
            torch.nn.modules.activation.ReLU: 'relu',
            torch.nn.modules.activation.Tanh: 'tanh',
            torch.nn.modules.activation.GELU: 'gelu'
        }
        
        _config = {
            'ks': self.ks_list,
            'num_features': self.num_features,
            'hidden_size': self.hidden_size,
            'activation': _activations.get(type(self.encoder_activation), str(self.encoder_activation)),
        }
        
        with open(Path(save_dir)/f"{name}_config.json", 'w') as fb:
            json.dump(_config, fb)
        log.info("    Configuration file saved.")
        return

class NeuralAdmixture():
    """
    Neural Admixture class.

    Args:
        k (int): Number of components for clustering.
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Learning rate for optimization.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        seed (int): Random seed for reproducibility.
        num_gpus (int): Number of GPUs available for training.
        master (bool): Indicates if the current process is the master process (used for logging/output control in multi-GPU settings).
        pack2bit (Any): Encoding used for compressing data in 2 bit format.
        supervised_loss_weight (Optional[float]): Weight of the supervised loss component (if using supervised training). Defaults to None.
    """
    def __init__(self, k: int, epochs: int, batch_size: int, learning_rate: float, device: torch.device, seed: int, num_gpus: int,
                master: bool, pack2bit, min_k: int, max_k: int, supervised_loss_weight: Optional[float]=100):
        """
        Initializes the NeuralAdmixture class with training parameters and settings.

        Args:
            k (int): Number of components for clustering.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
            learning_rate (float): Learning rate for optimization.
            device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
            seed (int): Random seed for reproducibility.
            num_gpus (int): Number of GPUs available for training.
            master (bool): Indicates if the current process is the master process (used for logging/output control in multi-GPU settings).
            pack2bit (Any): Encoding used for compressing data in 2 bit format.
            supervised_loss_weight (Optional[float]): Weight of the supervised loss component (if using supervised training). Defaults to None.
        """
        super(NeuralAdmixture, self).__init__()
        
        # Model configuration:
        self.k = k
        self.min_k = min_k
        self.max_k = max_k
        
        if k is not None:
            self.ks_list = [self.k]
        else:
            self.ks_list = list(range(self.min_k, self.max_k + 1))
        
        self.num_gpus = num_gpus
        self.device = device
        self.master = master
        
        # Random seed configuration
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
                
        # Training configuration:
        self.epochs = epochs
        self.batch_size = batch_size//self.num_gpus if self.num_gpus>0 else batch_size
        self.loss_function = torch.nn.BCELoss(reduction='sum').to(device)
        self.lr = learning_rate
        
        # Supervised version:
        self.supervised_loss_weight = supervised_loss_weight
        self.loss_function_supervised = torch.nn.CrossEntropyLoss(reduction='sum')
        
        #Pack2bit function
        self.pack2bit = pack2bit
        
    def initialize_model(self, P: torch.Tensor, hidden_size: int, num_features: int, V: torch.Tensor, ks_list: List) -> None:
        """
        Initializes the Q_P model and sets up distributed training if applicable.

        Args:
            P (torch.Tensor): Tensor representing the initial P matrix (e.g., allele frequencies).
            hidden_size (int): Number of units in the hidden layer of the encoder.
            num_features (int): Dimensionality of the input features.
            k (int): Number of components or clusters.
            V (torch.Tensor): PCA projection matrix used to reduce input dimensionality.

        Returns:
            None
        """
        self.base_model = Q_P(hidden_size, num_features, V, P, ks_list).to(self.device)
        if self.device.type == 'cuda':
            self.model = torch.compile(self.base_model)
        if self.num_gpus > 1 and torch.distributed.is_initialized():
            local_rank = torch.distributed.get_rank()
            self.model = torch.nn.parallel.DistributedDataParallel(self.base_model, device_ids=[local_rank], 
                                                                output_device=[local_rank], find_unused_parameters=False)
            self.raw_model = self.model.module
        else:
            self.model = self.base_model
            self.raw_model = self.base_model
                
    def launch_training(self, P: torch.Tensor, data: torch.Tensor, hidden_size:int, num_features:int, 
                        V: torch.Tensor, M: int, N: int, pops: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
        """
        Launches the training process, which includes two distinct phases and a final inference step to compute Q.

        Args:
            P (torch.Tensor): Initial tensor for the P matrix.
            data (torch.Tensor): Input data matrix (e.g., genotype matrix).
            hidden_size (int): Size of the hidden layer in the encoder.
            num_features (int): Number of input features (e.g., SNPs).
            k (int): Number of latent components or population clusters.
            V (torch.Tensor): PCA projection matrix used for dimensionality reduction.
            M (int): Number of SNPs (columns) in the dataset.
            N (int): Number of individuals (rows) in the dataset.
            y (Optional[torch.Tensor]): Optional labels for supervised loss (if available).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]: A tuple containing:
                - Trained P matrix.
                - Inferred Q matrix.
                - Trained raw model (Q_P instance).
        """ 
        #SETUP:
        self.M = M
        self.N = N
        torch.set_float32_matmul_precision('medium')
        torch.set_flush_denormal(True)
        self.initialize_model(P, hidden_size, num_features, V, self.ks_list)
        if pops is None:
            pops = torch.zeros(data.size(0), device=self.device)
            run_epoch = self._run_epoch
        else:
            run_epoch = self._run_epoch_supervised
            
        #TRAINING:
        if self.master:
            log.info("")
            log.info("    Starting training...")
            log.info("")
        self.optimizer = self.raw_model.create_custom_adam(device=self.device, lr=self.lr)
        dataloader = dataloader_admixture(data, self.batch_size, self.num_gpus, self.seed, self.generator, pops, shuffle=True)
        for epoch in tqdm(range(self.epochs), desc="Epochs", file=sys.stderr):
            run_epoch(epoch, dataloader)

        #INFERENCE OF Q's:
        self.raw_model.return_func = self.raw_model._return_infer
        batch_size_inference_Q = min(data.shape[0], 1024)
        self.model.eval()
        Qs = [torch.tensor([], device=self.device) for _ in self.ks_list]
        with torch.inference_mode():
            dataloader = dataloader_admixture(data, batch_size_inference_Q, 1 if self.num_gpus >= 1 else 0, self.seed, self.generator, pops, shuffle=False)
            for x_step, _ in dataloader:
                if self.pack2bit is not None:
                    unpacked_step = torch.empty((x_step.shape[0], self.M), dtype=torch.uint8, device=self.device)
                    self.pack2bit.unpack2bit_gpu_to_gpu(x_step, unpacked_step)
                    probs, _ = self.model(unpacked_step)
                else:
                    probs, _ = self.model(x_step)
                for i in range(len(self.ks_list)):
                    Qs[i]= torch.cat((Qs[i], probs[i]), dim=0)

        if self.master:
            log.info("")
            log.info("    Training finished!")
            log.info("")
            
        #RETURN OUTPUT:
        self.display_divergences(self.k)
        return self.process_results(Qs)

    def _run_epoch(self, epoch, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training.
        
        Args:
            epoch (int): Number of current epoch.
            dataloader (Dataloader): Dataloader
        """
        loss_acc = 0
        for x_step, _ in dataloader:
            if self.pack2bit is not None:
                unpacked_step = torch.empty((x_step.shape[0], self.M), dtype=torch.uint8, device=self.device)
                self.pack2bit.unpack2bit_gpu_to_gpu(x_step, unpacked_step)
                loss = self._run_step(unpacked_step)
            else:
                loss = self._run_step(x_step)
            loss.backward()
            self.optimizer.step()
            self.raw_model.restrict_P()
            
            loss_acc += loss.item()
        
        if epoch%5==0:
            log.info(f"            Loss in epoch {epoch:3d} on device {self.device} is {loss_acc:,.0f}")
        
    def _run_step(self, x_step: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step.

        Args:
            x_step (torch.Tensor): Batch of X data.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        recs, x_step = self.model(x_step)
        loss = sum((self.loss_function(rec, x_step) for rec in recs[0]))
        return loss
    
    def _run_epoch_supervised(self, epoch, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training (supervised version).
        
        Args:
            epoch (int): Number of current epoch.
            dataloader (Dataloader): Dataloader.
        """
        loss_acc = 0
        for x_step, pops_step in dataloader:
            if self.pack2bit is not None:
                unpacked_step = torch.empty((x_step.shape[0], self.M), dtype=torch.uint8, device=self.device)
                self.pack2bit.unpack2bit_gpu_to_gpu(x_step, unpacked_step)
                loss = self._run_step_supervised(unpacked_step, pops_step)
            else:
                loss = self._run_step_supervised(x_step, pops_step)
            
            loss.backward()
            self.optimizer.step()
            self.raw_model.restrict_P()
            
            loss_acc += loss.item()
        
        if epoch%2==0:
            log.info(f"            Loss in epoch {epoch:3d} on device {self.device} is {int(loss_acc):,.0f}")
            
    def _run_step_supervised(self, x_step: torch.Tensor, pops_step: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step.

        Args:
            X (torch.Tensor): Batch of X data.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        out, x_step = self.model(x_step)
        loss = self.loss_function(out[0][0], x_step)
        loss += self.supervised_loss_weight*self.loss_function_supervised(out[1][0], pops_step)
        return loss
    
    def display_divergences(self, k) -> None:
        """
        Displays pairwise Fst divergences between estimated populations.

        Args:
            k (int): Number of populations (K) used in the model.

        Details:
            - The function calculates and prints Hudson's Fst for each pair
            of estimated populations, providing a measure of genetic divergence.
        """
        if self.master:
            for i, k in enumerate(self.ks_list):
                dec = self.raw_model.decoders.decoders[i].weight.data
                header = '\t'.join([f'Pop{p}' for p in range(k - 1)])
                
                log.info("    Results:")
                log.info(f'\n            Fst divergences between estimated populations: (K = {k})')
                log.info("")
                log.info(f'                \t{header}')
                log.info('            Pop0')
                
                for j in range(1, k):
                    output = f'            Pop{j}'
                    pop2 = dec[:, j]
                    
                    for l in range(j):
                        pop1 = dec[:, l]
                        fst = self._hudsons_fst(pop1, pop2)
                        output += f"\t{fst:0.3f}"

                    log.info(output)
                
                log.info("\n")

    def process_results(self, Qs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
        """
        Processes and logs final results after training.

        Args:
            data (torch.Tensor): Original data tensor.
            Q (torch.Tensor): Learned Q matrix (assignments to populations).

        Returns:
            Tuple: Processed population matrix (P), Q matrix, and the raw model.

        Details:
            - Computes and logs the log-likelihood of the model given the data.
        """
        if self.master:
            Ps = [dec.weight.data.detach().cpu().numpy() for dec in self.raw_model.decoders.decoders]
            Qs = [Q.cpu().numpy() for Q in Qs]
        else:
            Ps, Qs = [], []
        return Qs, Ps, self.raw_model
    
    @staticmethod
    def _hudsons_fst(pop1: torch.Tensor, pop2: torch.Tensor) -> float:
        """
        Computes Hudson's Fst between two populations.

        Args:
            pop1 (torch.Tensor): Frequencies for population 1.
            pop2 (torch.Tensor): Frequencies for population 2.

        Returns:
            float: Hudson's Fst value.

        Formula:
            Fst = mean((p1 - p2)^2) / mean(p1 * (1-p2) + p2 * (1-p1))
        """
        try:
            num = torch.mean((pop1 - pop2) ** 2)
            den = torch.mean(pop1 * (1 - pop2) + pop2 * (1 - pop1)) + 1e-7
            return (num / den).item()
        except Exception as e:
            log.info(f"            Error computing Hudson's Fst: {e}")
            return float('nan')
