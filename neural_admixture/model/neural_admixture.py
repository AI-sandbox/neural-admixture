import logging
import sys
import json
import torch

from pathlib import Path
from typing import Callable, Optional, Tuple
from tqdm.auto import tqdm

from ..src.loaders import dataloader_admixture
from ..src.utils_c import pack2bit

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

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
    def __init__(self, hidden_size: int, num_features: int, k: int, activation: torch.nn.Module, P: torch.Tensor=None, V: torch.Tensor=None, 
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
        
        self.V = V
        
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
        X = X.float() / 2
        X = torch.where(X == 1.5, 0.0, X)
        
        X_pca = X@self.V
        X_pca = self.batch_norm(X_pca)
        hid_states = self.common_encoder(X_pca)
        probs = self.softmax(hid_states)
        return self.return_func(probs), X

    @torch.no_grad()
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
                device_tensors: torch.device, master: bool, num_cpus: int, supervised_loss_weight: Optional[float]=None):
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
        self.pin = True #True if 'gpu' in device_tensors.type else False
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
        
    def initialize_model(self, p_tensor: torch.Tensor, hidden_size: int, num_features: int, 
                         k: int, activation: torch.nn.Module, V: torch.Tensor) -> None:
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
        self.base_model = Q_P(hidden_size, num_features, k, activation, p_tensor, V).to(self.device)
        if self.device.type == 'cuda':
            self.model = torch.compile(self.base_model)
        if self.num_gpus > 1 and torch.distributed.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(self.base_model, device_ids=[self.device], 
                                                                output_device=[self.device], find_unused_parameters=False)
            self.raw_model = self.model.module
        else:
            self.model = self.base_model
            self.raw_model = self.base_model
                
    def launch_training(self, P: torch.Tensor, data: torch.Tensor, hidden_size:int, num_features:int, 
                        k:int, activation: torch.nn.Module, V: torch.Tensor, M: int, N: int,
                        y: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
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
        self.M = M
        self.N = N
        torch.set_float32_matmul_precision('medium')
        torch.set_flush_denormal(True)
        self.initialize_model(P, hidden_size, num_features, k, activation, V)
        if y is None:
            y = torch.zeros(data.size(0), device='cpu')
            run_epoch = self._run_epoch
        else:
            run_epoch = self._run_epoch_supervised
            
        #TRAINING:
        if self.master:
            log.info("")
            log.info("    Starting training...")
            log.info("")
        self.optimizer = self.raw_model.create_custom_adam(self.lr)
        dataloader = dataloader_admixture(data, self.batch_size, self.num_gpus, self.seed, self.generator, self.pin, y, self.num_cpus, shuffle=True, num_workers=4)
        for epoch in tqdm(range(self.epochs), desc="Epochs", file=sys.stderr):
            run_epoch(epoch, dataloader)

        #INFERENCE OF Q's:
        self.raw_model.return_func = self.raw_model._return_infer
        batch_size_inference_Q = min(data.shape[0], 5000)
        self.model.eval()
        Q = torch.tensor([], device=self.device)
        with torch.inference_mode():
            dataloader = dataloader_admixture(data, batch_size_inference_Q, 1 if self.num_gpus >= 1 else 0, self.seed, self.generator, self.pin, y,
                                            self.num_cpus, shuffle=False, num_workers=4)
            for x_step in dataloader:
                unpacked_step = torch.empty((x_step.shape[0], self.M), dtype=torch.uint8, device=self.device)
                pack2bit.unpack2bit_gpu_to_gpu(x_step, unpacked_step)
                out, _ = self.model(unpacked_step)
                Q = torch.cat((Q, out), dim=0)
        if self.num_gpus>1:
            torch.distributed.broadcast(Q, src=0)

        if self.master:
            log.info("")
            log.info("    Training finished!")
            log.info("")
            
        #RETURN OUTPUT:
        self.display_divergences(self.k)
        return self.process_results(data, Q)

    def _run_epoch(self, epoch, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training for Phase 2.
        
        Args:
            dataloader (Dataloader): Dataloader of phase 2.
        """
        loss_acc = 0
        for x_step in dataloader:
            unpacked_step = torch.empty((x_step.shape[0], self.M), dtype=torch.uint8, device=self.device)
            pack2bit.unpack2bit_gpu_to_gpu(x_step, unpacked_step)
            loss = self._run_step(unpacked_step)
            loss.backward()
            self.optimizer.step()
            self.raw_model.restrict_P()
            
            loss_acc += loss.item()
        
        if epoch%2==0:
            log.info(f"            Loss in epoch {epoch:3d} on device {self.device} is {loss_acc:,.0f}")
        
    def _run_step(self, x_step: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step for Phase 2.

        Args:
            X (torch.Tensor): Batch of X data.
            input_step (torch.Tensor): Batch of input data.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        recs, x_step = self.model(x_step)
        loss = self.loss_function(recs[0], x_step)
        return loss
    
    def _run_epoch_supervised(self, epoch, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training for Phase 2 (supervised version).
        
        Args:
            dataloader (Dataloader): Dataloader of phase 2.
        """
        loss_acc = 0
        for X, input_step, y in dataloader:
            X = X.to(self.device, non_blocking=True)
            input_step = input_step.to(self.device, non_blocking=True)
            
            loss = self._run_step_supervised(X, input_step, y)
            loss.backward()
            self.optimizer.step()
            self.raw_model.restrict_P()
            
            loss_acc += loss.item()
        
        if epoch%2==0:
            log.info(f"            Loss in epoch {epoch:3d} on device {self.device} is {int(loss_acc):,.0f}")
            
    def _run_step_supervised(self, X: torch.Tensor,  input_step: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step for Phase 2.

        Args:
            X (torch.Tensor): Batch of X data.
            input_step (torch.Tensor): Batch of input data.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        recs, probs = self.model(input_step)
        loss = self.loss_function(recs, X)
        mask = y > -1
        loss += self.supervised_loss_weight*self.loss_function_supervised(probs[mask], y[mask])
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

    def process_results(self, data: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
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
        self._loglikelihood(Q, self.raw_model.P.data.detach().T, data, self.master, self.M, self.device)
        
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
        
    @staticmethod
    def _loglikelihood(Q: torch.Tensor, P: torch.Tensor, data: torch.Tensor,
                      master: bool, M: int, device: torch.device, eps: float = 1e-7) -> None:
        """Compute deviance for a single K using PyTorch tensors in batches of 2048 samples.

        Args:
            Q (torch.Tensor): Matrix Q (shape N x K).
            P (torch.Tensor): Matrix P (shape K x M).
            data (torch.Tensor): original data (shape N x M).
            master (bool): flag to indicate if this is the master process.
            eps (float, optional): epsilon term to avoid numerical errors. Defaults to 1e-7.
        """
        if master:
            batch_size = 2048
            total_loglikelihood = torch.tensor(0.0, device=Q.device)
            num_samples = Q.size(0)
            num_batches = (num_samples + batch_size - 1) // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                Q_batch = Q[start_idx:end_idx, :]
                
                data_batch = data[start_idx:end_idx, :]
                unpacked_step = torch.empty((data_batch.shape[0], M), dtype=torch.uint8, device=device)
                pack2bit.unpack2bit_gpu_to_gpu(data_batch, unpacked_step)
                
                unpacked_step = torch.clamp(unpacked_step, eps, 2 - eps)
                rec_batch = torch.clamp(torch.matmul(Q_batch, P), eps, 1 - eps)

                loglikelihood_batch = unpacked_step * torch.log(rec_batch) + (2 - unpacked_step) * torch.log1p(-rec_batch)

                total_loglikelihood += torch.sum(loglikelihood_batch)

            result = total_loglikelihood
            log.info(f"    Total Log likelihood: {result.item():,.0f}")
            log.info("\n")

            return