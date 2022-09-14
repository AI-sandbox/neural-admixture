import dask.array as da
import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import time
import torch
from collections.abc import Iterable
from dask_ml.decomposition import IncrementalPCA as DaskIncrementalPCA
from sklearn.cluster import KMeans
from typing import Union
from pathlib import Path
from py_pcha import PCHA


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def pca_plot(X_pca: np.ndarray, path: str) -> None:
    """Helper function to render a PCA plot

    Args:
        X_pca (np.ndarray): projected data
        path (str): output file path
    """
    plt.figure(figsize=(15,10))
    plt.scatter(X_pca[:,0], X_pca[:,1], s=.9, c='black')
    plt.xticks([])
    plt.yticks([])
    plt.title('Training data projected onto first two components')
    plt.savefig(path)
    log.info('Plot rendered.')
    return

class PCKMeansInitialization(object):
    """PCKMeans initialization
    """
    @classmethod
    def get_decoder_init(cls, X: da.core.Array, K: Union[Iterable, int], path: Union[str, None], run_name: str, n_components: int, seed: int, batch_size: int) -> torch.Tensor:
        """Get decoder initialization weights using PCKMeans

        Args:
            X (da.core.Array): 2D genotype array
            K (Union[Iterable, int]): number of ancestries to run the algorithm on. If an iterable, then assumes multihead version will be run
            path (Union[str, None]): output path
            run_name (str): name characterizing the experiment
            n_components (int): number of components to compute via incremental PCA
            seed (int): RNG seed
            batch_size (int, optional): batch size for incremental PCA. Defaults to 400.

        Returns:
            torch.Tensor: decoder initialization
        """
        log.info('Running PC-KMeans initialization...')
        np.random.seed(seed)
        t0 = time.time()
        try:
            if path is not None:
                with open(path, 'rb') as fb:
                    pca_obj = pickle.load(fb)
                log.info('PCA loaded.')
                if pca_obj.n_components_ != n_components:
                    raise FileNotFoundError
            else:
                raise FileNotFoundError
        except FileNotFoundError as fnfe:
            log.info(f"{n_components}D PCA object not found. Performing IncrementalPCA...")
            pca_obj = DaskIncrementalPCA(n_components=n_components, random_state=42, batch_size=batch_size)
            pca_obj.fit(X)
            if path is not None:
                with open(path, 'wb') as fb:
                    pickle.dump(pca_obj, fb)
        except Exception as e:
            raise e
        assert pca_obj.n_features_ == X.shape[1], 'Computed PCA and training data do not have same number of SNPs' 
        log.info('Projecting data...')
        X_pca = pca_obj.transform(X).compute()
        log.info('Running KMeans on projected data...')
        if isinstance(K, Iterable):
            k_means_objs = [KMeans(n_clusters=i, random_state=42, n_init=10, max_iter=10).fit(X_pca) for i in K]
            centers = np.concatenate([obj.cluster_centers_ for obj in k_means_objs])
            P_init = torch.as_tensor(pca_obj.inverse_transform(centers).compute(), dtype=torch.float32).view(sum(K), -1)
        else:
            k_means_obj = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=10).fit(X_pca)
            P_init = torch.as_tensor(pca_obj.inverse_transform(k_means_obj.cluster_centers_).compute(), dtype=torch.float32).view(K, -1)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        log.info('Rendering PCA plot...')
        try:
            if path is not None:
                plot_save_path = Path(path).parent/f"{run_name}_training_pca.png"
                pca_plot(X_pca, plot_save_path)
        except Exception as e:
            log.warn(f'Could not render PCA plot: {e}')
            log.info('Resuming...')
        return P_init

class PCArchetypal(object):
    """PCArchetypal initialization
    """
    @classmethod
    def get_decoder_init(cls, X: da.core.Array, K: Union[Iterable, int], path: str, run_name: str, n_components: int, seed: int, batch_size: int):
        """Get decoder initialization weights using PCArchetypal

        Args:
            X (da.core.Array): 2D genotype array
            K (Union[Iterable, int]): number of ancestries to run the algorithm on. If an iterable, then assumes multihead version will be run
            path (str): output path
            run_name (str): name characterizing the experiment
            n_components (int): number of components to compute via incremental PCA
            seed (int): RNG seed
            batch_size (int, optional): batch size for incremental PCA. Defaults to 400.

        Returns:
            torch.Tensor: decoder initialization
        """
        log.info('Running PCArchetypal initialization...')
        np.random.seed(seed)
        t0 = time.time()
        try:
            if path is not None:
                with open(path, 'rb') as fb:
                    pca_obj = pickle.load(fb)
                log.info('PCA loaded.')
                if pca_obj.n_components_ != n_components:
                    raise FileNotFoundError
            else:
                raise FileNotFoundError
        except FileNotFoundError as fnfe:
            log.info(f'{n_components}D PCA object not found. Performing PCA...')
            pca_obj = DaskIncrementalPCA(n_components=n_components, random_state=42, batch_size=batch_size)
            pca_obj.fit(X)
            if path is not None:
                with open(path, 'wb') as fb:
                    pickle.dump(pca_obj, fb)
        except Exception as e:
            raise e
        assert pca_obj.n_features_ == X.shape[1], 'Computed PCA and training data do not have same number of SNPs'
        log.info(f'Projecting data to {n_components} dimensions...')
        X_proj = pca_obj.transform(X).compute()
        if not isinstance(K, Iterable):
            K = [K]
        log.info(f'Executing archetypal analysis on projected data...')
        archs = np.concatenate([np.array(PCHA(X_proj.T, noc=k, delta=0.001)[0].T) for k in K])
        log.info('Backtransforming archetypes...')
        P_init = torch.as_tensor(pca_obj.inverse_transform(archs).compute(), dtype=torch.float32)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        try:
            if path is not None:
                plot_save_path = Path(path).parent/f"{run_name}_training_pca.png"
                pca_plot(X_proj, plot_save_path)
        except Exception as e:
            log.warn(f'Could not render PCA plot: {e}')
            log.info('Resuming...')
        return P_init


class SupervisedInitialization(object):
    """Supervised initialization
    """
    @classmethod
    def get_decoder_init(cls, X, y, K):
        log.info('Running supervised initialization...')
        assert y is not None, 'Ground truth ancestries needed for supervised mode'
        if len(K) > 1:
            raise NotImplementedError("Supervised mode is only available for single-head architectures at the moment.")
        t0 = time.time()
        k = K[0]
        ancestry_dict = {anc: idx for idx, anc in enumerate(sorted(np.unique([a for a in y if a != '-'])))}
        assert len(ancestry_dict) == k, f'Number of ancestries in training ground truth ({len(ancestry_dict)}) is not equal to the value of K ({k})'
        ancestry_dict['-'] = -1
        to_idx_mapper = np.vectorize(lambda x: ancestry_dict[x])
        # Do not take into account samples with missing labels
        y_num = to_idx_mapper(y[:])
        mask = y_num > -1
        masked_y_num = y_num[mask]
        X_masked = X[mask,:]
        P_init = torch.as_tensor(np.vstack([X_masked[masked_y_num==idx,:].mean(axis=0).compute() for idx in range(k)]), dtype=torch.float32)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        return P_init


class PretrainedInitialization(object):
    """Pretrained initialization
    """
    @classmethod
    def get_decoder_init(cls, X, K, path):
        log.info('Fetching pretrained weights...')
        if len(K) > 1:
            raise NotImplementedError("Pretrained mode is only supported for single-head runs.")
        # Loads standard ADMIXTURE output format
        P_init = torch.as_tensor(1-np.genfromtxt(path, delimiter=' ').T, dtype=torch.float32)
        assert P_init.shape[0] == K[0], 'Input P is not coherent with the value of K'
        log.info('Weights fetched.')
        return P_init
