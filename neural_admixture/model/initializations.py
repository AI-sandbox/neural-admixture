import logging
import numpy as np
import sys

import torch
import time
import dask.array as da
import dask

from sklearn.mixture import GaussianMixture as GaussianMixture
from typing import Tuple


from .em_adam import optimize_parameters
from .refinement import refine_Q_P, loglikelihood

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
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
        
        P, Q = refine_Q_P(P, Q, data, device, num_cpus, num_epochs=5, patience=2)
        
        logl = loglikelihood(Q, P, data)
        log.info("")
        log.info(f"    Final log-likelihood: {logl:.1f}")  

        return P, Q