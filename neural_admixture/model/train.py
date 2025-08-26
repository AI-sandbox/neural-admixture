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

log = logging.getLogger(__name__)

def train(epochs: int, batch_size: int, learning_rate: float, K: int, seed: int,
        data: torch.Tensor, device: torch.device, num_gpus: int, hidden_size: int, 
        master: bool, V: np.ndarray, pops : np.ndarray, min_k: int=None, max_k: int=None, n_components: int=None) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
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
        
        N, M = data.shape
    
        if master:
            
            if pops is None:
                data = data.numpy()
                # PCA:
                X_pca = np.zeros((N, n_components), dtype=np.float32)
                for i in range(0, N, 1024):
                    end_idx = min(i + 1024, N)
                    batch = data[i:end_idx, :].astype(np.float32)/2
                    X_pca[i:end_idx] = batch@V.T
                X_pca = X_pca.astype('float64')
                
                # GMM:
                log.info("")
                log.info("    Running Gaussian Mixture in PCA subspace...")
                log.info("")
                if K is not None:
                    gmm = GaussianMixture(n_components=K, n_init=5, init_params='k-means++', tol=1e-4, covariance_type='full', max_iter=100, random_state=seed)        
                    gmm.fit(X_pca)
                    P = np.clip((gmm.means_@V), 5e-6, 1 - 5e-6)
                    del gmm
                else:
                    gmm_objs = [GaussianMixture(n_components=K, n_init=5, init_params='k-means++', tol=1e-4, covariance_type='full', max_iter=100, random_state=seed).fit(X_pca) for K in range(min_k, max_k + 1)]
                    P = np.concatenate([np.clip((obj.means_@V), 5e-6, 1 - 5e-6) for obj in gmm_objs], axis=0)
                    del gmm_objs
                del X_pca
                
                data = torch.as_tensor(data, dtype=torch.uint8, device='cpu' if device.type != 'mps' else 'mps')
            else:
                data = data.numpy()
                
                log.info("")
                log.info("    Running Supervised Mode...")
                log.info("")
                ancestry_dict = {anc: idx for idx, anc in enumerate(sorted(np.unique([a for a in pops])))}
                assert len(ancestry_dict) == K, f'Number of ancestries in training ground truth ({len(ancestry_dict)}) is not equal to the value of K ({K})'
                to_idx_mapper = np.vectorize(lambda x: ancestry_dict[x])
                y_num = to_idx_mapper(pops[:])     
                P = np.vstack([data[y_num == idx, :].astype(np.float32).mean(axis=0) for idx in range(K)])
                                
                data = torch.as_tensor(data, dtype=torch.uint8, device='cpu' if device.type != 'mps' else 'mps')

        if torch.distributed.is_initialized():    
            dist.barrier()
        
        if num_gpus>1:
            if master:
                P_init = torch.as_tensor(P, dtype=torch.float32, device=device).contiguous()
                V = torch.as_tensor(V.T, dtype=torch.float32, device=device).contiguous()
                if pops is not None:
                    pops = torch.as_tensor(y_num, dtype=torch.int64, device=device)
            else:
                if K is not None:
                    P_init = torch.empty((K, M), dtype=torch.float32, device=device)
                    if pops is not None:
                        pops = torch.empty(len(pops), dtype=torch.int64, device=device)
                else:
                    total_K = sum(range(min_k, max_k + 1))
                    P_init = torch.empty((total_K, M), dtype=torch.float32, device=device)
                V = torch.empty((M, n_components), dtype=torch.float32, device=device)
                
            if master:
                log.info("    Broadcasting to all GPUs...")
            if pops is not None:
                dist.broadcast(pops, src=0)
            dist.broadcast(P_init, src=0)
            dist.broadcast(V, src=0)
            dist.barrier()
            if master:
                log.info("    Finished broadcasting!")
        else:
            P_init = torch.as_tensor(P, dtype=torch.float32, device=device).contiguous()
            V = torch.as_tensor(V.T, dtype=torch.float32, device=device).contiguous()
            if pops is not None:
                pops = torch.as_tensor(y_num, dtype=torch.int64, device=device)

        if num_gpus>0 and device.type != 'mps':
            packed_data = torch.empty((N, (M + 3) // 4), dtype=torch.uint8, device=device)
            from neural_admixture import __file__ as installation_dir
            from pathlib import Path
            source_path = os.path.abspath(f"{Path(installation_dir).parent}/src/utils_c/pack2bit.cu")
            pack2bit = load(name="pack2bit", sources=[source_path], verbose=True)
            pack2bit.pack2bit_cpu_to_gpu(data, packed_data)
        else:
            pack2bit = None
            packed_data = data
        
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, master, pack2bit, min_k, max_k)
        Qs, Ps, model = model.launch_training(P_init, packed_data, hidden_size, V.shape[1], V, M, N, pops)
        
        if master:
            data = data.cpu().numpy()
            if K is not None:
                P = np.ascontiguousarray(Ps[0].astype(np.float64))
                Q = np.ascontiguousarray(Qs[0].astype(np.float64))
                logl = utils.loglikelihood(data, P, Q, K)
                log.info(f"    Log-likelihood: {logl:2f}.")
            else:
                for i, K in enumerate(range(min_k, max_k + 1)):
                    P = np.ascontiguousarray(Ps[i].astype(np.float64))
                    Q = np.ascontiguousarray(Qs[i].astype(np.float64))
                    logl = utils.loglikelihood(data, P, Q, K)
                    log.info(f"    Log-likelihood for K={K}: {logl:2f}.")
        del data
        del packed_data
        return Ps, Qs, model
