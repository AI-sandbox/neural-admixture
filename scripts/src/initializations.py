import logging
import numpy as np
import os
import time
import torch
from collections.abc import Iterable
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class KMeansInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, k, minibatch, use_logit, batch_size=200, seed=42):
        weights_path = '/mnt/gpid08/users/albert.dominguez/weights/chr1/minibatch_kmeans_logit_2gens_10iters_down6.pt'
        if minibatch and use_logit:
            if os.path.exists(weights_path):
                log.info('Loading k-means cluster centroids from precomputed tensor.')
                return torch.load(weights_path)
        log.info('Computing k-Means cluster centroids...')
        if not isinstance(k, Iterable):
            if minibatch:
                k_means_obj = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=seed, n_init=1, max_iter=10).fit(X)
            else:
                k_means_obj = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=1).fit(X)
            if not use_logit:
                return torch.tensor(k_means_obj.cluster_centers_).float()
            P_init = torch.clamp(torch.tensor(k_means_obj.cluster_centers_).float(), min=1e-4, max=1-1e-4) 
            del k_means_obj
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                raise Exception
            return P_init
        else:
            log.info(f'Running {len(k)} algorithms (multihead)...')
            k_means_objs = [None]*len(k)
            if minibatch:
                k_means_objs = [MiniBatchKMeans(n_clusters=i, batch_size=batch_size, random_state=seed, n_init=1, max_iter=1, compute_labels=False).fit(X)
                                for i in k]
            else:
                k_means_objs = [KMeans(n_clusters=i, random_state=42, n_init=1, max_iter=1, compute_labels=False).fit(X) for i in k]
            if not use_logit:
                return torch.cat([torch.tensor(obj.cluster_centers_) for obj in k_means_objs], axis=0).float()
            P_init = torch.clamp(torch.cat([torch.tensor(obj.cluster_centers_) for obj in k_means_objs], axis=0).float(),
                                 min=1e-4, max=1-1e-4)
            del k_means_objs
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                raise Exception
            torch.save(P_init, weights_path)
            log.info('All k-means run successfully. Initialization was saved.')
            return P_init


class KMeansPlusPlusInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, k, seed=42):
        weights_path = '/mnt/gpid08/users/albert.dominguez/weights/chr1/kmeans++_2gens_down6.pt'
        if os.path.exists(weights_path):
            log.info('Loading k-means++ centroids from precomputed tensor.')
            return torch.load(weights_path)
        log.info('Computing k-Means++ centroids...')
        X_np = np.array(X[:])
        if not isinstance(k, Iterable):
            centers, _ = kmeans_plusplus(X_np, n_clusters=k, random_state=seed)
            P_init = torch.clamp(torch.tensor(centers).float(), min=1e-4, max=1-1e-4) 
            del centers, X_np
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                raise Exception
            return P_init
        else:
            log.info(f'Running {len(k)} algorithms (multihead)...')
            centers = [kmeans_plusplus(X_np, n_clusters=i, random_state=seed)[0]
                       for i in k]
            P_init = torch.clamp(torch.cat([torch.tensor(obj) for obj in centers], axis=0).float(),
                                 min=1e-4, max=1-1e-4)
            del centers, X_np
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                raise Exception
            torch.save(P_init, weights_path)
            log.info('All k-means++ run successfully. Initialization was saved.')
            return P_init

class RandomMeanInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, K):
        X_mean = torch.tensor(np.mean(X, axis=0)).unsqueeze(1)
        P_init = (torch.bernoulli(X_mean.repeat(1, K))-0.5).T.float()
        return P_init

class SNPsMeanInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, K):
        X_mean = torch.tensor(np.mean(X, axis=0), dtype=torch.float)
        return X_mean.repeat(K, 1)

class BinomialInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, K):
        log.info(f'Running binomial initialization...')
        P_inits = [torch.bernoulli(torch.tensor([0.]*X.shape[1]), p=0.1) for k in range(sum(K))]
        return torch.logit(torch.vstack(P_inits), eps=1e-4)
