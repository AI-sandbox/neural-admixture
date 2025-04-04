import logging
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.mixture import GaussianMixture as GaussianMixture
from typing import Tuple
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from collections import deque

from .em_adam import optimize_parameters

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
        
        import time
        import dask.array as da
        import dask
        t0 = time.time()
        
        with dask.config.set({"random.seed": seed}):
            data_dask = da.from_array(data.T, chunks=(768, 768))
            if np.any(data == 9):
                data_dask = da.where(data_dask == 9, 0, data_dask)
            log.info("    Data dask matrix created")
            _, _, V = da.linalg.svd_compressed(data_dask, k=K, compute=True, n_power_iter=0, iterator='power', n_oversamples=10)
        log.info("    V matrix created")
        V=V.compute().astype(np.float32)
        t1 = time.time()
        
        log.info(f"    Time spent in svd: {t1-t0:.2f}s")
        
        # PCA:
        X_pca = np.zeros((N, K), dtype=np.float32)
        for i in range(0, N, 1024):
            end_idx = min(i + 1024, N)
            batch = data[:, i:end_idx].T.astype(np.float32)/2
            X_pca[i:end_idx] = batch@V.T
        
        # GMM:
        log.info("")
        log.info("    Running Gaussian Mixture in PCA subspace...")
        log.info("")
        gmm = GaussianMixture(n_components=K, n_init=5, init_params='k-means++', tol=1e-4, covariance_type='full', max_iter=100, random_state=seed)        
        gmm.fit(X_pca)
        
        # ADAM EM:
        P = np.ascontiguousarray(np.clip((gmm.means_@V).T, 1e-5, 1 - 1e-5), dtype=np.float32)
        Q = np.clip(gmm.predict_proba(X_pca).astype(np.float32), 1e-5, 1 - 1e-5)

        log.info("    Adam expectation maximization running...")
        log.info("")
        P, Q = optimize_parameters(data, P, Q, seed)
        
        # REFINEMENT:
        log.info("    Refinement algorithm running...")
        log.info("")
        P = torch.as_tensor(P.T, dtype=torch.float32, device=device)
        Q = torch.as_tensor(Q, dtype=torch.float32, device=device)
        data = torch.as_tensor(data.T, dtype=torch.uint8)
        
        P, Q = train_model(P, Q, data, device, num_epochs=100, patience=5)
        
        logl = loglikelihood(Q, P, data)
        log.info(f"    Final log-likelihood: {logl:.1f}")  

        return P, Q

class AdmixtureDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return idx, self.data[idx].float() / 2

class AdmixtureOptimizer(nn.Module):
    def __init__(self, init_P, init_Q):
        super().__init__()
        self.P = nn.Parameter(init_P)
        self.Q = nn.Parameter(init_Q)
    
    def forward(self, indices):
        Q_batch = self.Q[indices]
        R = torch.mm(Q_batch, self.P)
        return torch.clamp(R, 0.0, 1.0)

def train_model(P, Q, data, device, num_epochs=100, patience=5):
    dataset = AdmixtureDataset(data)
    batch_size = 512
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model = AdmixtureOptimizer(P, Q).to(device)
    criterion = nn.BCELoss()
    
    best_loss = float('inf')
    best_P = model.P.clone().detach()
    best_Q = model.Q.clone().detach()
    recent_losses = deque(maxlen=patience)
    
    for epoch in tqdm(range(num_epochs), desc="Epochs", file=sys.stderr):
        loss_acc = 0.0
        
        if epoch % 2 == 0:
            model.Q.requires_grad_(False)
            model.P.requires_grad_(True)
            optimizer = optim.Adam([model.P], lr=5e-6, fused=True)
        else:
            model.Q.requires_grad_(True)
            model.P.requires_grad_(False)
            optimizer = optim.Adam([model.Q], lr=5e-6, fused=True)
        
        for indices, X_batch in dataloader:
            indices = indices.to(device)
            X_batch = X_batch.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            R = model(indices)
            loss = criterion(R, X_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.P.clamp_(0.0, 1.0)
                model.Q.clamp_(0.0, 1.0)
            
            loss_acc += loss.item()
        
        log.info(f"    Epoch {epoch+1}, Loss: {loss_acc}")
        
        if loss_acc < best_loss:
            best_loss = loss_acc
            best_P = model.P.clone().detach()
            best_Q = model.Q.clone().detach()
        
        recent_losses.append(loss_acc)
        
        if len(recent_losses) == patience and not any(loss < recent_losses[0] for loss in list(recent_losses)[1:]):
            log.info("")
            log.info("    Early stopping triggered")
            log.info("")
            break

    return best_P.cpu(), best_Q.cpu()

def loglikelihood(Q: torch.Tensor, P: torch.Tensor, data: torch.Tensor, batch_size: int = 512, eps: float = 1e-7) -> float:
    """Compute deviance for a single K using PyTorch tensors

    Args:
        Q (torch.Tensor): Matrix Q of shape (N, D).
        P (torch.Tensor): Matrix P of shape (D, M).
        data (torch.Tensor): Original data of shape (N, M).
        batch_size (int, optional): Size of each batch. Defaults to 64.
        eps (float, optional): Epsilon term to avoid numerical errors. Defaults to 1e-7.

    Returns:
        float: Computed log-likelihood value.
    """
    result = 0.0
    N = Q.shape[0]
    
    for init in range(0, N, batch_size):
        end = min(init + batch_size, N)
        Q_batch = Q[init:end]
        data_batch = data[init:end]
        
        rec_batch = torch.clamp(torch.matmul(Q_batch, P), eps, 1 - eps)
        data_batch = torch.clamp(data_batch, eps, 2 - eps)

        loglikelihood_batch = data_batch * torch.log(rec_batch) + (2 - data_batch) * torch.log1p(-rec_batch)
        
        result += torch.sum(loglikelihood_batch).item()
        
    return result