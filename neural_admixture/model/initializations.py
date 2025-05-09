import logging
import os
import numpy as np
import sys

import torch
import time
import dask.array as da
import dask
import torch.distributed as dist
from torch.utils.cpp_extension import load

from sklearn.mixture import GaussianMixture as GaussianMixture
from typing import Tuple

from .neural_admixture import NeuralAdmixture
from .em_adam import optimize_parameters

from ..src.utils_c import rsvd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logging.getLogger("distributed").setLevel(logging.WARNING)

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"

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

        N, M = data.shape

        #data = np.ascontiguousarray(data.T)
    
        if master:
            # SVD:
            V = randomized_svd_uint8_input(data, K, N, M)

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

        data = torch.as_tensor(data, dtype=torch.uint8, device='cpu')
        packed_data = torch.empty((N, (M + 3) // 4), dtype=torch.uint8, device=device)
        
        source_path = os.path.abspath("neural-admixture-dev/neural_admixture/src/utils_c/pack2bit.cu")
        pack2bit = load(name="pack2bit", sources=[source_path], verbose=True)
        pack2bit.pack2bit_cpu_to_gpu(data, packed_data)
        
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device, master, num_cpus, pack2bit)
        
        P, Q, _ = model.launch_training(P_init, packed_data, hidden_size, V.shape[1], K, activation, V, M, N)
        
        return P, Q
    
# -----------------------------------------------------------------------------
# High-level randomized SVD function
# -----------------------------------------------------------------------------

def randomized_svd_uint8_input(A_uint8, k, N, M, oversampling=10, power_iterations=4):
    """
    Randomized SVD para matrices uint8 de forma (n_features, m_samples).
    Retorna Vt_k de forma (k, m_samples).
    """
    k_prime = min(N, M, k + oversampling)

    total_start_time = time.time()
    log.info("1) Generando Ω y Y = A @ Ω...")
    Omega = np.random.randn(M, k_prime).astype(np.float32)
    Y = rsvd.multiply_A_omega(A_uint8, Omega)
    log.info(f"   Y.shape={Y.shape}, time={time.time() - total_start_time:.4f}s")

    if power_iterations > 0:
        iter_start = time.time()
        for _ in range(power_iterations):
            Q_y, _ = np.linalg.qr(Y, mode='reduced')    # (n, k_prime)
            Q_y = np.ascontiguousarray(Q_y.T)
            B_tmp = rsvd.multiply_QT_A(Q_y, A_uint8)      # (k_prime, m)
            B_tmp = np.ascontiguousarray(B_tmp.T)
            Y = rsvd.multiply_A_omega(A_uint8, B_tmp)     # (n, k_prime)
        log.info(f"   Power iterations time={time.time() - iter_start:.4f}s")

    log.info("2) QR de Y...")
    qr_start = time.time()
    Q, _ = np.linalg.qr(Y, mode='reduced')            # (n, k_prime)
    log.info(f"   Q.shape={Q.shape}, time={time.time() - qr_start:.4f}s")

    log.info("3) B = Qᵀ @ A...")
    b_start = time.time()
    Q = np.ascontiguousarray(Q.T)
    B = rsvd.multiply_QT_A(Q, A_uint8)                   # (k_prime, m)
    log.info(f"   B.shape={B.shape}, time={time.time() - b_start:.4f}s")

    log.info("4) SVD de B...")
    svd_start = time.time()
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    log.info(f"   SVD time={time.time() - svd_start:.4f}s")

    log.info(f"Total time SVD: {time.time() - total_start_time:.4f}s")
    return Vt[:k, :]
