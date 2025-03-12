import logging
import sys
import json
import torch

from pathlib import Path
from typing import Callable, Optional, Tuple
from tqdm.auto import tqdm

from ..src.loaders import dataloader_train, dataloader_inference

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def compile_if_cuda(func: Callable) -> Callable:
    """
    Compiles the given function with CUDA optimizations if a GPU is available.

    This decorator checks if CUDA (GPU support) is available. If so, it uses
    `torch.compile` with specific options for optimizing the function for execution
    on the GPU. If CUDA is not available, the function is returned as-is.

    Args:
        func (Callable): The function to potentially compile for CUDA.

    Returns:
        Callable: The compiled function if CUDA is available, otherwise the original function.
    """
    if torch.cuda.is_available():
        return torch.compile(func, options={"triton.cudagraphs": True}, fullgraph=True)
    return func

class Q_P(torch.nn.Module):
    """
    Q_P model.
    
    Args:
        hidden_size (int): The size of the hidden layer.
        num_features (int): The number of features in the input data.
        k (int): The number of output classes or components.
        activation (torch.nn.Module): The activation function to use in the encoder.
        P (Optional[torch.Tensor], optional): The P matrix to be optimized. Defaults to None.
        
    """
    def __init__(self, hidden_size: int, num_features: int, k: int, activation: torch.nn.Module, P: torch.Tensor=None,
                is_train: bool=True) -> None:
        """
        Initialize the Q_P module with the given parameters.

        Args:
            hidden_size (int): The size of the hidden layer.
            num_features (int): The number of features in the input data.
            k (int): The number of output classes or components.
            activation (torch.nn.Module): The activation function to use in the encoder.
            P (Optional[torch.Tensor], optional): The P matrix to be optimized. Defaults to None.
        """
        super(Q_P, self).__init__()
        self.k = k
        if P is not None:
            self.P = torch.nn.Parameter(P)
        
        self.num_features = num_features
        self.batch_norm = torch.nn.RMSNorm(self.num_features, eps=1e-8)
        self.encoder_activation = activation
        self.hidden_size = hidden_size
        self.common_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.num_features, self.hidden_size, bias=True),
            self.encoder_activation,
            torch.nn.Linear(self.hidden_size, self.k, bias=True)
        )
        self.softmax = torch.nn.Softmax(dim=1)
        
        if is_train:
            self.return_func = self._return_training
        else:
            self.return_func = self._return_infer
                
    def _return_training(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.clamp_(torch.nn.functional.linear(probs, self.P), 0, 1), probs
    
    def _return_infer(self, probs: torch.Tensor) -> torch.Tensor:
        return probs
     
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass with the given batch of input data.

        Args:
            X (torch.Tensor): A tensor of input data.

        Returns:
            torch.Tensor: The result of the forward pass, which is either probabilities or a transformed tensor clamped between 0 and 1.
        """
        X = self.batch_norm(X)
        hid_states = self.common_encoder(X)
        probs = self.softmax(hid_states)
        return self.return_func(probs)

    @torch.no_grad()
    @compile_if_cuda
    def restrict_P(self):
        """
        Restrict the values of P matrix within the range [0, 1].
        """
        self.P.clamp_(0., 1.)
    
    def create_custom_adam(self, lr: float=1e-5) -> torch.optim.Adam:
        """
        Creates a custom Adam optimizer with different learning rates for different phases.

        Args:
            lr (float): Learning rate for all parameters.

        Returns:
            optim.Adam: The Adam optimizer configured with the specified learning rates.
        """
        p = [
        {'params': self.P, 'lr': lr},
        {'params': self.common_encoder.parameters(), 'lr': lr},
        {'params': self.batch_norm.parameters(), 'lr': lr}
            ]
        return torch.optim.Adam(p, betas=[0.9, 0.95], fused=True)
    
    
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
            'k': self.k,
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
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
            seed (int): Random seed for reproducibility.
            num_gpus (int): Number of GPUs available for training.
            device_tensors (torch.device): Device for tensor data.
    """
    def __init__(self, k: int, epochs: int, batch_size: int, learning_rate: float, device: torch.device, seed: int, num_gpus: int,
                device_tensors: torch.device, master: bool, num_cpus: int):
        """
        Initializes the NeuralAdmixture class with training parameters and settings.

        Args:
            k (int): Number of components for clustering.
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            learning_rate (float): Learning rate for all parameters.
            device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
            seed (int): Random seed for reproducibility.
            num_gpus (int): Number of GPUs available for training.
            device_tensors (torch.device): Device for tensor data.
            master (bool): Wheter or not this process is the master for printing the output.
        """
        super(NeuralAdmixture, self).__init__()
        
        # Model configuration:
        self.k = k
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.device = device
        self.pin = True if 'cpu' in device_tensors.type else False
        self.master = master
        
        # Random seed configuration
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
                
        # Training configuration:
        self.epochs = epochs
        self.batch_size = batch_size//self.num_gpus if self.num_gpus>0 else batch_size
        self.loss_function = torch.nn.BCELoss(reduction='sum')
        self.lr = learning_rate
        
    def initialize_model(self, p_tensor: torch.Tensor, hidden_size: int, num_features: int, 
                         k: int, activation: torch.nn.Module) -> None:
        """
        Initializes the Q_P model and sets up distributed training if applicable.

        Args:
            p_tensor (torch.Tensor): P tensor to initialize the model.
            hidden_size (int): Hidden layer size for the encoder.
            num_features (int): Number of input features.
            k (int): Number of clusters or components.
            activation (torch.nn.Module): Activation function for the encoder layers.
            num_total_elements (int): Total number of elements in the dataset.

        Returns:
            None
        """
        self.base_model = Q_P(hidden_size, num_features, k, activation, p_tensor).to(self.device)
        if self.num_gpus > 1 and torch.distributed.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(self.base_model, device_ids=[self.device], 
                                                                output_device=[self.device], find_unused_parameters=False)
            self.raw_model = self.model.module
        else:
            self.model = self.base_model
            self.raw_model = self.base_model
                
    def launch_training(self, P: torch.Tensor, data: torch.Tensor, hidden_size:int, num_features:int, 
                        k:int, activation: torch.nn.Module, input: torch.Tensor, f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
        """
        Launches the training process, which includes two distinct phases and the inference step to compute Q.

        Args:
            Q (torch.Tensor): Initial tensor for Q matrix.
            P (torch.Tensor): Initial tensor for P matrix.
            data (torch.Tensor): Input data matrix.
            hidden_size (int): Size of the hidden layer in the model.
            num_features (int): Number of features in the input data.
            k (int): Number of latent components or clusters.
            activation (torch.nn.Module): Activation function for the model.
            input (torch.Tensor): Input tensor for phase 2 network.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]: Processed P, Q matrices and the raw model.
        """ 
        #SETUP:
        torch.set_float32_matmul_precision('medium')
        torch.set_flush_denormal(True)
        self.initialize_model(P, hidden_size, num_features, k, activation)

        #TRAINING:
        if self.master:
            log.info("")
            log.info("    Starting training...")
            log.info("")
        self.optimizer = self.raw_model.create_custom_adam(self.lr)
        dataloader = dataloader_train(data, input, f, self.batch_size, self.num_gpus, self.seed, self.generator, self.pin, self.num_cpus)
        for epoch in tqdm(range(self.epochs), desc="Epochs", file=sys.stderr):
            self._run_epoch(epoch, dataloader)

        #INFERENCE OF Q's:
        batch_size_inference_Q = min(data.shape[0], 5000)
        self.model.eval()
        Q = torch.tensor([], device=self.device)
        with torch.inference_mode():
            dataloader = dataloader_inference(input, batch_size_inference_Q, self.seed, self.generator, num_gpus=1 if self.num_gpus >= 1 else 0, 
                                              pin=self.pin, num_cpus=self.num_cpus)
            for input_step in dataloader:
                input_step = input_step.to(self.device)
                _, out = self.model(input_step)
                Q = torch.cat((Q, out), dim=0)
        if self.num_gpus>1:
            torch.distributed.broadcast(Q, src=0)

        if self.master:
            log.info("")
            log.info("    Training finished!")
            log.info("")
            
        #RETURN OUTPUT:
        self.display_divergences(self.k)
        return self.process_results(Q)

    def _run_epoch(self, epoch, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training.
        
        Args:
            dataloader (Dataloader): Dataloader.
        """
        loss_acc = 0
        for X, input_step in dataloader:
            X = X.to(self.device, non_blocking=self.pin)
            input_step = input_step.to(self.device, non_blocking=self.pin)
            
            loss = self._run_step(X, input_step)
            loss.backward()
            self.optimizer.step()
            self.raw_model.restrict_P()
            
            loss_acc += loss.item()
        
        if epoch%25==0:
            log.info(f"            Loss in epoch {epoch:3d} on device {self.device} is {loss_acc:,.0f}")
        
    def _run_step(self, X: torch.Tensor,  input_step: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step.

        Args:
            X (torch.Tensor): Batch of X data.
            input_step (torch.Tensor): Batch of input data.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        recs, _ = self.model(input_step)
        loss = self.loss_function(recs, X)
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
            dec = self.raw_model.P.data.detach().to('cpu')
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

    def process_results(self, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
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
        
        return self.raw_model.P.data.detach().cpu().numpy(), Q.cpu().numpy(), self.raw_model
    
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