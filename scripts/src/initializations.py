import logging
import numpy as np
import torch
from collections.abc import Iterable
from sklearn.cluster import KMeans, MiniBatchKMeans

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class KMeansInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, k, minibatch, use_logit, batch_size=200, seed=42):
        log.info('Getting k-Means cluster centroids...')
        if not isinstance(k, Iterable):
            if minibatch:
                k_means_obj = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=seed, n_init=1).fit(X)
            else:
                k_means_obj = KMeans(n_clusters=k, random_state=42, n_init=1).fit(X)
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
                k_means_objs = [MiniBatchKMeans(n_clusters=i, batch_size=batch_size, random_state=seed, n_init=1).fit(X)
                                for i in k]
            else:
                k_means_objs = [KMeans(n_clusters=i, random_state=42, n_init=1).fit(X) for i in k]
            if not use_logit:
                return torch.cat([torch.tensor(obj.cluster_centers_) for obj in k_means_objs], axis=0).float()
            P_init = torch.clamp(torch.cat([torch.tensor(obj.cluster_centers_) for obj in k_means_objs], axis=0).float(),
                                 min=1e-4, max=1-1e-4)
            del k_means_objs
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                raise Exception
            log.info('All k-means run successfully')
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