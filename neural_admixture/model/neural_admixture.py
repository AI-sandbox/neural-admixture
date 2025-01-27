import logging
import sys
import json
import torch

from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Optional, Tuple
from tqdm.auto import tqdm

from ..src.loaders import dataloader_P1, dataloader_P2, dataloader_inference

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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
            self.return_P2 = self._return_training
        else:
            self.return_P2 = self._return_infer
            
        self.dummy_param = torch.nn.Parameter(torch.zeros(1)) #remove find_unused_parameters warning in multi-gpu.
    
    def _return_training(self, probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.clamp(torch.nn.functional.linear(probs, self.P), 0, 1), probs
    
    def _return_infer(self, probs: torch.Tensor) -> torch.Tensor:
        return probs
     
    def forward(self, X: torch.Tensor, phase='P2', context_manager: Optional[torch.autocast] = nullcontext()) -> torch.Tensor:
        """
        Perform a forward pass with the given batch of input data.

        Args:
            X (torch.Tensor): A tensor of input data.
            phase (str, optional): The phase of processing ('P1' or 'P2'). Defaults to 'P2'.
            context_manager (Optional[torch.autocast], optional): A context manager for controlling execution flow, can be used for mixed-precision training.

        Returns:
            torch.Tensor: The result of the forward pass, which is either probabilities or a transformed tensor clamped between 0 and 1.
        """
        if phase == 'P2':
            X = self.batch_norm(X)
            hid_states = self.common_encoder(X)
            probs = self.softmax(hid_states)
            return self.return_P2(probs)
        else:
            with context_manager:
                return torch.clamp(torch.nn.functional.linear(X, self.P),0,1)

    @torch.no_grad()
    @compile_if_cuda
    def restrict_P(self):
        """
        Restrict the values of P matrix within the range [0, 1].
        """
        self.P.clamp_(0., 1.)
    
    def create_custom_adam(self, lr_P1_P: float=3e-4, lr_P2: float=1e-5, phase='P1') -> torch.optim.Adam:
        """
        Creates a custom Adam optimizer with different learning rates for different phases.

        Args:
            lr_P1_P (float, optional): Learning rate for phase 1 for the P parameters. Defaults to 3e-4.
            lr_P2 (float, optional): Learning rate for phase 2. Defaults to 1e-5.
            phase (str, optional): The current phase of training. Can be 'P1' or 'P2'. Defaults to 'P1'.

        Returns:
            optim.Adam: The Adam optimizer configured with the specified learning rates.
        """
        if phase == 'P1':
            p = [
                {'params': self.P, 'lr': lr_P1_P},
                ]
        else:
            p = [
            {'params': self.P, 'lr': lr_P2},
            {'params': self.common_encoder.parameters(), 'lr': lr_P2},
            {'params': self.batch_norm.parameters(), 'lr': lr_P2}
                ]
        return torch.optim.Adam(p, betas=[0.9, 0.95], fused=True)
    
    def freeze(self) -> None:
        """
        Freezes the parameters of the common encoder and batch normalization layers
        by setting their `requires_grad` to False, which prevents them from being updated during training.
        """
        for param in self.common_encoder.parameters():
            param.requires_grad = False
        for param in self.batch_norm.parameters():
            param.requires_grad = False
        return
    
    def unfreeze(self) -> None:
        """
        Unfreezes the parameters of the common encoder and batch normalization layers
        by setting their `requires_grad` to True, allowing them to be updated during training.
        """
        for param in self.common_encoder.parameters():
            param.requires_grad = True
        for param in self.batch_norm.parameters():
            param.requires_grad = True
        return
    
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
        log.info('Configuration file saved.')
        return

class NeuralAdmixture():
    """
    Neural Admixture class.
    
    Args:
            k (int): Number of components for clustering.
            epochs_P1 (int): Number of epochs for phase 1.
            epochs_P2 (int): Number of epochs for phase 2.
            batch_size_P1 (int): Batch size for phase 1.
            batch_size_P2 (int): Batch size for phase 2.
            device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
            seed (int): Random seed for reproducibility.
            num_gpus (int): Number of GPUs available for training.
            learning_rate_P1_P (float): Learning rate for P in phase 1.
            learning_rate_P2 (float): Learning rate for all parameters in phase 2.
            device_tensors (torch.device): Device for tensor data.
    """
    def __init__(self, k: int, epochs_P1: int, epochs_P2: int, batch_size_P1: int, batch_size_P2: int, 
                 device: torch.device, seed: int, num_gpus: int, learning_rate_P1_P: float, 
                 learning_rate_P2: float, device_tensors: torch.device, master: bool, num_cpus: int,
                 supervised_loss_weight: Optional[float]=None):
        """
        Initializes the NeuralAdmixture class with training parameters and settings.

        Args:
            k (int): Number of components for clustering.
            epochs_P1 (int): Number of epochs for phase 1.
            epochs_P2 (int): Number of epochs for phase 2.
            batch_size_P1 (int): Batch size for phase 1.
            batch_size_P2 (int): Batch size for phase 2.
            device (torch.device): Device for computation (e.g., 'cuda' or 'cpu').
            seed (int): Random seed for reproducibility.
            num_gpus (int): Number of GPUs available for training.
            learning_rate_P1_P (float): Learning rate for P in phase 1.
            learning_rate_P2 (float): Learning rate for all parameters in phase 2.
            device_tensors (torch.device): Device for tensor data.
            master (bool): Wheter or not this process is the master for printing the output.
        """
        super(NeuralAdmixture, self).__init__()
        
        # Model configuration:
        self.k = k
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.device = device
        self.device_type = 'cuda' if 'cuda' in self.device.type else 'cpu'
        self.pin = True if 'cpu' in device_tensors.type else False
        self.dtype = torch.bfloat16 if 'cuda' in self.device.type else torch.float32
        self.master = master
        
        # Context manager for mixed precision:
        self.context_manager = torch.autocast(device_type=self.device_type, dtype=self.dtype) if self.num_gpus>0 else nullcontext()

        # Random seed configuration
        self.seed = seed
        self.generator = torch.Generator().manual_seed(self.seed)
        
        # Accumulation steps for gradient scaling (if GPUs are used):
        self.accumulation_steps = 8 / self.num_gpus if self.num_gpus>0 else 8
        
        # Training configuration:
        self.epochs_P1 = epochs_P1
        self.epochs_P2 = epochs_P2
        self.batch_size_P1 = batch_size_P1//self.num_gpus if self.num_gpus>0 else batch_size_P1
        self.batch_size_P2 = batch_size_P2//self.num_gpus if self.num_gpus>0 else batch_size_P2
        self.loss_function_P1 = torch.nn.BCELoss(reduction='mean')
        self.loss_function_P2 = torch.nn.BCELoss(reduction='sum')
        self.lr_P1_P = learning_rate_P1_P
        self.lr_P2 = learning_rate_P2
        
        # Supervised version:
        self.supervised_loss_weight = supervised_loss_weight
        self.loss_function_supervised = torch.nn.CrossEntropyLoss(reduction='sum')
        
    def initialize_model(self, p_tensor: torch.Tensor, hidden_size: int, num_features: int, 
                         k: int, activation: torch.nn.Module, num_total_elements: int) -> None:
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
                                                                output_device=[self.device], find_unused_parameters=True)
            self.raw_model = self.model.module
        else:
            self.model = self.base_model
            self.raw_model = self.base_model
        
        self.num_batches_P1 = num_total_elements // (self.batch_size_P1)
        self.num_batches_P2 = num_total_elements // (self.batch_size_P2)
        
    def launch_training(self, P: torch.Tensor, data: torch.Tensor, hidden_size:int, num_features:int, 
                        k:int, activation: torch.nn.Module, input: torch.Tensor, Q: Optional[torch.Tensor]=None, 
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
        torch.set_float32_matmul_precision('medium')
        self.initialize_model(P, hidden_size, num_features, k, activation, data.shape[0])
        
        if y is None:
            y = torch.zeros(data.size(0))
        
        #PHASE 1:
        if Q is not None: #not supervised
            self.optimizer = self.raw_model.create_custom_adam(self.lr_P1_P, phase='P1')
            self.raw_model.freeze()
            dataloader = dataloader_P1(data, Q, self.batch_size_P1, self.num_gpus, self.seed, self.generator, self.pin, self.num_cpus)
            for _ in tqdm(range(self.epochs_P1), desc="Epochs"):
                self._run_epoch_P1(dataloader)
            self.raw_model.unfreeze()
                                
        #PHASE 2:
        data = data.float()
        self.optimizer = self.raw_model.create_custom_adam(lr_P2=self.lr_P2, phase='P2')
        dataloader = dataloader_P2(data, input, self.batch_size_P2, self.num_gpus, self.seed, self.generator, self.pin, y, self.num_cpus)

        run_epoch = self._run_epoch_P2_supervised if Q is None else self._run_epoch_P2
        for _ in tqdm(range(self.epochs_P2), desc="Epochs"):
            run_epoch(dataloader)

        #INFERENCE OF Q's:
        batch_size_inference_Q = min(data.shape[0], 5000)
        self.model.eval()
        Q = torch.tensor([], device=self.device)
        with torch.inference_mode():
            dataloader = dataloader_inference(input, batch_size_inference_Q, self.seed, self.generator, num_gpus=1 if self.num_gpus >= 1 else 0, 
                                              pin=self.pin, num_cpus=self.num_cpus)
            for input_step in dataloader:
                input_step = input_step.to(self.device)
                _, out = self.model(input_step, 'P2')
                Q = torch.cat((Q, out), dim=0)
        if self.num_gpus>1:
            torch.distributed.broadcast(Q, src=0)

        #RETURN OUTPUT
        self.display_divergences(self.k)
        return self.process_results(data, Q)

    def _run_epoch_P1(self, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training for Phase 1.
        
        Args:
            dataloader (Dataloader): Dataloader of phase 1.
        """
        for batch_idx, (q_tensor_batch, target_batch) in enumerate(dataloader):
            q_tensor_batch = q_tensor_batch.to(self.device, non_blocking=self.pin)
            target_batch = target_batch.to(self.device, non_blocking=self.pin)
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx + 1 == self.num_batches_P1:   
                loss = self._run_step_P1(q_tensor_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)  
                self.raw_model.restrict_P()
            else:
                if self.num_gpus > 1:
                    with self.model.no_sync():
                        loss = self._run_step_P1(q_tensor_batch, target_batch)
                        loss.backward()
                else:
                    loss = self._run_step_P1(q_tensor_batch, target_batch)
                    loss.backward()
                    
    def _run_step_P1(self, q_tensor_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step for Phase 1.

        Args:
            q_tensor_batch (torch.Tensor): Batch of Q tensor.
            target_batch (torch.Tensor): Corresponding targets for the batch.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        loss = self.loss_function_P1(self.model(q_tensor_batch, 'P1', self.context_manager), target_batch)
        loss = loss/self.accumulation_steps
        return loss

    def _run_epoch_P2(self, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training for Phase 2.
        
        Args:
            dataloader (Dataloader): Dataloader of phase 2.
        """
        for X, input_step, _ in dataloader:
            X = X.to(self.device, non_blocking=self.pin)
            input_step = input_step.to(self.device, non_blocking=self.pin)
            
            loss = self._run_step_P2(X, input_step)
            loss.backward()
            self.optimizer.step()
            self.raw_model.restrict_P()
        
    def _run_step_P2(self, X: torch.Tensor,  input_step: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step for Phase 2.

        Args:
            X (torch.Tensor): Batch of X data.
            input_step (torch.Tensor): Batch of input data.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        recs, _ = self.model(input_step, 'P2')
        loss = self.loss_function_P2(recs, X)
        return loss
    
    def _run_epoch_P2_supervised(self, dataloader: torch.utils.data.DataLoader):
        """
        Executes one epoch of training for Phase 2 (supervised version).
        
        Args:
            dataloader (Dataloader): Dataloader of phase 2.
        """
        for X, input_step, y in dataloader:
            X = X.to(self.device, non_blocking=self.pin)
            input_step = input_step.to(self.device, non_blocking=self.pin)
            
            loss = self._run_step_P2_supervised(X, input_step, y)
            loss.backward()
            self.optimizer.step()
            self.raw_model.restrict_P()
            
    def _run_step_P2_supervised(self, X: torch.Tensor,  input_step: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Executes one training step for Phase 2.

        Args:
            X (torch.Tensor): Batch of X data.
            input_step (torch.Tensor): Batch of input data.

        Returns:
            torch.Tensor: Computed loss for the batch.
        """
        self.optimizer.zero_grad(set_to_none=True)
        recs, probs = self.model(input_step, phase='P2')
        loss = self.loss_function_P2(recs, X)
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

        Note:
            Only runs on the first GPU ('cuda:0') or CPU.
        """
        if self.master:
            dec = self.raw_model.P.data.detach().to('cpu')
            header = '\t'.join([f'Pop{p}' for p in range(k - 1)])
            print(f'\nFst divergences between estimated populations: (K = {k})')
            print(f'\t{header}')
            print('Pop0')
            for j in range(1, k):
                print(f'Pop{j}', end='')
                pop2 = dec[:, j]
                for l in range(j):
                    pop1 = dec[:, l]
                    fst = self._hudsons_fst(pop1, pop2)
                    print(f"\t{fst:0.3f}", end="" if l != j - 1 else '\n')
            print("\n")
        return
    
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
        self._loglikelihood(Q.cpu(), self.raw_model.P.data.detach().cpu().T, data.cpu(), self.device, self.master)
        
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
            # Element-wise operations using PyTorch
            num = torch.mean((pop1 - pop2) ** 2)
            den = torch.mean(pop1 * (1 - pop2) + pop2 * (1 - pop1)) + 1e-7
            return (num / den).item()
        except Exception as e:
            print(f"Error computing Hudson's Fst: {e}")
            return float('nan')
        
    @staticmethod
    def _loglikelihood(Q: torch.Tensor, P: torch.Tensor, data: torch.Tensor, device: torch.device, 
                      master: bool, eps: float = 1e-7, reduction: str = "sum") -> None:
        """Compute deviance for a single K using PyTorch tensors

        Args:
            Q (torch.Tensor): Matrix Q.
            P (torch.Tensor): Matrix P.
            data (torch.Tensor): original data.
            mask (torch.Tensor): mask tensor.
            eps (float, optional): epsilon term to avoid numerical errors. Defaults to 1e-7.
            reduction (str, optional): reduction method. Should be either 'mean' or 'sum'. Defaults to 'sum'.
        """
        if master:
            assert reduction in ("mean", "sum"), "reduction argument should be either 'mean' or 'sum'"
            
            rec = torch.clamp(torch.matmul(Q, P), eps, 1 - eps)
            rec_2 = torch.clamp(torch.matmul(Q, (1 - P)), eps, 1 - eps)
            data = torch.clamp(data * 2, eps, 2 - eps)

            loglikelihood = data * torch.log(rec) + (2 - data) * torch.log(rec_2)
            
            if reduction == "sum":
                result = torch.sum(loglikelihood)
            elif reduction == "mean":
                result = torch.mean(loglikelihood)
            else:
                raise ValueError("Unknown reduction")
            
            log.info(f"Log likelihood: {result.item()}")
            return