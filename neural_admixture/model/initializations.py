import logging
import numpy as np
import sys
import time
import torch
from scipy import sparse

from typing import Tuple

from .neural_admixture import NeuralAdmixture

from .em_quasi import optimize_parameters

from ..src.utils_c import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def determine_device_for_tensors(data_shape: tuple, K: int, device: torch.device, memory_threshold: float = 0.9) -> torch.device:
    """
    Determine if tensors can fit in GPU memory and return appropriate device.
    
    Args:
        data_shape: Shape of the input data tensor
        K: Number of components/clusters
        device: Current torch device
        memory_threshold: Fraction of available GPU memory to use (default: 0.9)
        
    Returns:
        torch.device: Device to use for tensors ('cuda' if they fit, 'cpu' if they don't)
    """
    def bytes_to_human_readable(bytes_value: int) -> str:
        """Convert bytes to human readable string (GB/MB)"""
        gb = bytes_value / (1024**3)
        if gb >= 1:
            return f"{gb:.2f} GB"
        mb = bytes_value / (1024**2)
        return f"{mb:.2f} MB"

    def calculate_tensor_memory(shape, dtype=torch.float32) -> int:
        """Calculate memory required for tensor of given shape and dtype"""
        num_elements = np.prod(shape)
        bytes_per_element = dtype.itemsize
        return num_elements * bytes_per_element
    
    if 'cuda' in device.type:
        available_gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
        
        memory_required = {
            'P': calculate_tensor_memory((data_shape[1], K)),
            'Q': calculate_tensor_memory((data_shape[0], K)),
            'data': calculate_tensor_memory(data_shape, 
                                        dtype=torch.bfloat16 if 'cuda' in device.type else torch.float32)
        }
        
        total_memory_required = sum(memory_required.values())
        fits_in_gpu = total_memory_required <= (available_gpu_memory * memory_threshold)
        device_tensors = device if fits_in_gpu else torch.device('cpu')
        
        if str(device) == 'cuda:0':
            log.info(f"    Tensors stored in {('GPU' if fits_in_gpu else 'CPU')} because there are "
            f"{bytes_to_human_readable(available_gpu_memory)} available in GPU and "
            f"tensors occupy {bytes_to_human_readable(total_memory_required)}")
    else:
        device_tensors = device
        
    return device_tensors


class RandomInitialization(object):
    """
    Class to initialize a neural admixture model using Gaussian Mixture Models (GMM).
    """
    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int,
                        n_components: int, data: np.ndarray, device: torch.device, num_gpus: int, hidden_size: int, 
                        activation: torch.nn.Module, master: bool, num_cpus: int, has_missing: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
        """
        Initializes P and Q matrices and trains a neural admixture model using GMM.

        Args:
            epochs (int): Number of epochs
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            K (int): Number of components (clusters).
            seed (int): Random seed for reproducibility.
            init_path (Path): Path to store PCA initialization.
            name (str): Name identifier for the model.
            n_components (int): Number of PCA components.
            data (np.ndarray): Input data array (samples x features).
            device (torch.device): Device for computation (e.g., CPU or GPU).
            num_gpus (int): Number of GPUs available.
            hidden_size (int): Hidden layer size for the model.
            activation (torch.nn.Module): Activation function.
            master (bool): Wheter or not this process is the master for printing the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]: Initialized P matrix, Q matrix, and trained model.
        """
        if master:
            log.info("    Running Random initialization...")

        M, N = data.shape
        device_tensors = determine_device_for_tensors((N, M), K, device)
        if has_missing:
            f = np.zeros(M, dtype=np.float32)
            utils.estimateMean(data, f)
            f = torch.as_tensor(f.T, dtype=torch.float32, device=device_tensors)
        else:
            f = None
        
        X_subspace = sparse_random_projection(data, n_components, device_tensors, master)
        indices = np.random.choice(N, K, replace=False)
        P_init = torch.as_tensor(data[:, indices], dtype=torch.float32, device=device).contiguous() / 2
        data = torch.as_tensor(data.T, dtype=torch.uint8, device=device_tensors)
        
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus)
        
        P, Q, raw_model = model.launch_training(P_init, data, hidden_size, X_subspace.shape[1], K, activation, X_subspace, f)
        
        data = np.ascontiguousarray(data.T.cpu(), dtype=np.uint8)
        P = np.ascontiguousarray(P,  dtype=np.float64)
        Q = np.ascontiguousarray(Q,  dtype=np.float64)
        
        optimize_parameters(data, P, Q, seed)
        
        return P, Q, raw_model

def sparse_random_projection(data, dim, device, master):
    """
    Implements an efficient random projection for matrices, using:
    - Dense projection for small/medium matrices
    - Sparse projection for large matrices

    Parameters:
    - data: NumPy matrix of shape (M, N) [SNPs x samples]
    - dim: dimension of the projected subspace
    - device: torch device ('cuda' or 'cpu')

    Returns:
    - X_subspace_tensor: torch tensor with projected data
    """
    M, N = data.shape
    
    if M <= 100000 or N <= 50000:  # Small or medium matrix → dense projection
        if master:
            log.info("    Projection on a dense matrix...")
        projection_matrix = np.random.randn(M, dim) / np.sqrt(M)
        X_subspace = np.zeros((N, dim), dtype=np.float32)
        for i in range(N):  
            X_subspace[i, :] = data[:, i].astype(np.float32) @ projection_matrix  
    
    else:  # Large matrix → sparse projection
        if master:
            log.info("    Projection on a sparse matrix...")
        density = min(3.0 / np.sqrt(M), 1.0)
        s = 1.0 / np.sqrt(density)
        projection_matrix = sparse.random(M, dim, density=density, 
                                          data_rvs=lambda n: np.random.choice([-s, s], size=n),
                                          format='csr')
        batch_size = min(10000, N)  # Adjust based on available memory
        X_subspace = np.zeros((N, dim), dtype=np.float32)
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch = data[:, i:end_idx].T.astype(np.float32)
            X_subspace[i:end_idx] = batch @ projection_matrix
    
    # Convert to PyTorch tensor
    X_subspace_tensor = torch.tensor(X_subspace, dtype=torch.float32, device=device)
    
    return X_subspace_tensor
