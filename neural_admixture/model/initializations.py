import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time
import torch
from collections.abc import Iterable
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.decomposition import PCA, TruncatedSVD
from py_pcha import PCHA


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def pca_plot(X_pca, path):
    plt.figure(figsize=(15,10))
    plt.scatter(X_pca[:,0], X_pca[:,1], s=.9, c='black')
    plt.xticks([])
    plt.yticks([])
    plt.title('Training data projected onto first two components')
    plt.savefig(path)
    log.info('Plot rendered.')
    return

class PCKMeansInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, K, path, run_name, n_components):
        log.info('Running PC-KMeans initialization...')
        t0 = time.time()
        try:
            with open(path, 'rb') as fb:
                pca_obj = pickle.load(fb)
            log.info('PCA loaded.')
            if pca_obj.n_components_ != n_components:
                raise FileNotFoundError
        except FileNotFoundError as fnfe:
            log.info(f'{n_components}D PCA object not found. Performing PCA...')
            pca_obj = PCA(n_components=n_components, random_state=42)
            pca_obj.fit(X)
            with open(path, 'wb') as fb:
                pickle.dump(pca_obj, fb)
        except Exception as e:
            raise e
        assert pca_obj.n_features_ == X.shape[1], 'Computed PCA and training data do not have same number of SNPs' 
        log.info('Projecting data...')
        X_pca = pca_obj.transform(X)
        log.info('Running KMeans on projected data...')
        if isinstance(K, Iterable):
            k_means_objs = [KMeans(n_clusters=i, random_state=42, n_init=10, max_iter=10).fit(X_pca) for i in K]
            centers = np.concatenate([obj.cluster_centers_ for obj in k_means_objs])
            P_init = torch.as_tensor(pca_obj.inverse_transform(centers), dtype=torch.float32).view(sum(K), -1)
        else:
            k_means_obj = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=10).fit(X_tsvd)
            P_init = torch.as_tensor(pca_obj.inverse_transform(k_means_obj.cluster_centers_), dtype=torch.float32).view(K, -1)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        log.info('Rendering PCA plot...')
        try:
            save_root = '/'.join(path.split('/')[:-1]) if len(path.split('/')) > 1 else '.'
            plot_save_path = f'{save_root}/{run_name}_training_pca.png'
            pca_plot(X_pca, plot_save_path)
        except Exception as e:
            log.warn(f'Could not render PCA plot: {e}')
            log.info('Resuming...')
        return P_init

class PCArchetypal(object):
    @classmethod
    def get_decoder_init(cls, X, K, path, run_name, n_components, seed):
        log.info('Running ArchetypalPCA initialization...')
        np.random.seed(seed)
        t0 = time.time()
        try:
            with open(path, 'rb') as fb:
                pca_obj = pickle.load(fb)
            log.info('PCA loaded.')
            if pca_obj.n_components_ != n_components:
                raise FileNotFoundError
        except FileNotFoundError as fnfe:
            log.info(f'{n_components}D PCA object not found. Performing PCA...')
            pca_obj = PCA(n_components=n_components, random_state=42)
            pca_obj.fit(X[:])
            with open(path, 'wb') as fb:
                pickle.dump(pca_obj, fb)
        except Exception as e:
            raise e
        assert pca_obj.n_features_ == X.shape[1], 'Computed PCA and training data do not have same number of SNPs'
        log.info(f'Projecting data to {n_components} dimensions...')
        X_proj = pca_obj.transform(X[:])
        if not isinstance(K, Iterable):
            K = [K]
        log.info(f'Executing archetypal analysis on projected data...')
        archs = np.concatenate([np.array(PCHA(X_proj.T, noc=k, delta=0.001)[0].T) for k in K])
        log.info('Backtransforming archetypes...')
        P_init = torch.as_tensor(pca_obj.inverse_transform(archs), dtype=torch.float32)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        return P_init


class SupervisedInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, y, K):
        log.info('Running supervised initialization...')
        assert y is not None, 'Ground truth ancestries needed for supervised mode'
        if len(K) > 1:
            raise NotImplementedError
        t0 = time.time()
        k = K[0]
        ancestry_dict = {anc: idx for idx, anc in enumerate(sorted(np.unique(y)))}
        assert len(ancestry_dict) == k, f'Number of ancestries in training ground truth ({len(ancestry_dict)}) is not equal to the value of K ({k})'
        to_idx_mapper = np.vectorize(lambda x: ancestry_dict[x])
        y_num = to_idx_mapper(y[:])
        X_mem = X[:,:]
        P_init = torch.as_tensor(np.vstack([X_mem[y_num==idx,:].mean(axis=0) for idx in range(k)]), dtype=torch.float32)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        return P_init