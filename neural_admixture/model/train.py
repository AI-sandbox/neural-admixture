import logging
import os
import numpy as np
import sys
import torch
import torch.distributed as dist

from torch.utils.cpp_extension import load
from sklearn.mixture import GaussianMixture as GaussianMixture
from typing import Tuple
from .neural_admixture import NeuralAdmixture
from ..src.utils_c import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logging.getLogger("distributed").setLevel(logging.WARNING)

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

log = logging.getLogger(__name__)

def train(epochs: int, batch_size: int, learning_rate: float, K: int, seed: int,
        data: torch.Tensor, device: torch.device, num_gpus: int, hidden_size: int, 
        master: bool, V: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
        """
        Initializes P and Q matrices and trains a neural admixture model using GMM.

        Args:
            epochs (int): Number of epochs
            batch_size (int): Batch size.
            learning_rate (float): Learning rate.
            K (int): Number of components (clusters).
            seed (int): Random seed for reproducibility.
            data (torch.Tensor): Input data array (samples x features).
            device (torch.device): Device for computation (e.g., CPU or GPU).
            num_gpus (int): Number of GPUs available.
            hidden_size (int): Hidden layer size for the model.
            master (bool): Wheter or not this process is the master for printing the output.
            V (np.ndarray): V matrix for PCA.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]: Initialized P matrix, Q matrix, and trained model.
        """
        if master:
            log.info("    Running Random initialization...")

        N, M = data.shape
    
        if master:
            # SVD:
            data = data.numpy()

            # PCA:
            X_pca = np.zeros((N, K), dtype=np.float32)
            for i in range(0, N, 1024):
                end_idx = min(i + 1024, N)
                batch = data[i:end_idx, :].astype(np.float32)/2
                X_pca[i:end_idx] = batch@V.T
            
            # GMM:
            log.info("")
            log.info("    Running Gaussian Mixture in PCA subspace...")
            log.info("")
            gmm = GaussianMixture(n_components=K, n_init=5, init_params='k-means++', tol=1e-4, covariance_type='full', max_iter=100, random_state=seed)        
            gmm.fit(X_pca)
            del X_pca
            
            P = np.ascontiguousarray(np.clip((gmm.means_@V).T, 5e-6, 1 - 5e-6), dtype=np.float32)
            del gmm
            
            data = torch.as_tensor(data, dtype=torch.uint8, device='cpu')

        if torch.distributed.is_initialized():    
            dist.barrier()
        if num_gpus>1:
            if master:
                P_init = torch.as_tensor(P, dtype=torch.float32, device=device).contiguous()
                V = torch.as_tensor(V.T, dtype=torch.float32, device=device).contiguous()
            else:
                P_init = torch.empty((M, K), dtype=torch.float32, device=device)
                V = torch.empty((M, K), dtype=torch.float32, device=device)
            
            if master:
                log.info("    Broadcasting to all GPUs...")
            dist.broadcast(P_init, src=0)
            dist.broadcast(V, src=0)
            dist.barrier()
            if master:
                log.info("    Finished broadcasting!")

        else:
            P_init = torch.as_tensor(P, dtype=torch.float32, device=device).contiguous()
            V = torch.as_tensor(V.T, dtype=torch.float32, device=device).contiguous()

        if num_gpus>0:
            packed_data = torch.empty((N, (M + 3) // 4), dtype=torch.uint8, device=device)
            source_path = os.path.abspath("neural-admixture-dev/neural_admixture/src/utils_c/pack2bit.cu")
            pack2bit = load(name="pack2bit", sources=[source_path], verbose=True)
            pack2bit.pack2bit_cpu_to_gpu(data, packed_data)
        else:
            pack2bit=None
            packed_data = data
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, master, pack2bit)
        
        P, Q, model = model.launch_training(P_init, packed_data, hidden_size, V.shape[1], K, V, M, N)
        
        if master:
            data = data.numpy()
            logl = utils.loglikelihood(data, P, Q, K)
            log.info(f"    Log-likelihood: {logl:2f}.")
        return P, Q, model