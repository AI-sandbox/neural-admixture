import logging
import numpy as np
import sys

import torch
import time
import dask.array as da
import dask
import torch.distributed as dist

from sklearn.mixture import GaussianMixture as GaussianMixture
from typing import Tuple

from .neural_admixture import NeuralAdmixture
from .em_adam import optimize_parameters
from ..src.utils_c import pack2bit

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logging.getLogger("distributed").setLevel(logging.WARNING)

log = logging.getLogger(__name__)

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
        
        t0 = time.time()
        
        # SVD:
        if master:
            dask.config.set({"optimization.fuse.ave-width": 5})
            with dask.config.set({"random.seed": seed}):
                data_dask = da.from_array(data.T, chunks=(N, 100_000), asarray=False)
                if np.any(data == 9):
                    data_dask = da.where(data_dask == 9, 0, data_dask)
                data_dask = data_dask.persist()
                _, _, V = da.linalg.svd_compressed(data_dask, k=K, seed=da.random.RandomState(RandomState=np.random.RandomState), compute=True) 
            V = V.compute()
            del data_dask
            t1 = time.time()
            log.info(f"    Time spent in SVD: {t1-t0:.2f}s")
            
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
            del X_pca
            
            P = np.ascontiguousarray(np.clip((gmm.means_@V).T, 5e-6, 1 - 5e-6), dtype=np.float32)
            del gmm
        
        if torch.distributed.is_initialized():    
            dist.barrier()
        if num_gpus>1:
            if master:
                P_init = torch.as_tensor(P, dtype=torch.float32, device=device).contiguous()
                V = torch.as_tensor(V.T, dtype=torch.float32, device=device).contiguous()
            else:
                P_init = torch.empty((M, K), dtype=torch.float32, device=device)
                V = torch.empty((M, K), dtype=torch.float32, device=device)
            
            log.info("    Broadcasting to all GPUs...")
            dist.broadcast(P_init, src=0)
            dist.broadcast(V, src=0)
            dist.barrier()
            log.info("    Finished broadcasting!")

        else:
            P_init = torch.as_tensor(P, dtype=torch.float32, device=device).contiguous()
            V = torch.as_tensor(V.T, dtype=torch.float32, device=device).contiguous()

        data = torch.as_tensor(data.T, dtype=torch.uint8, device='cpu')
        packed_data = torch.empty((N, (M + 3) // 4), dtype=torch.uint8, device=device)
        pack2bit.pack2bit_cpu_to_gpu(data, packed_data)
        
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device, master, num_cpus)
        
        P, Q, _ = model.launch_training(P_init, packed_data, hidden_size, V.shape[1], K, activation, V, M, N)
        
        return P, Q