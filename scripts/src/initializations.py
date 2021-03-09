import logging
import numpy as np
import os
import time
import torch
from collections.abc import Iterable
from sklearn.cluster import KMeans, MiniBatchKMeans

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class KMeansInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, k, minibatch, use_logit, batch_size=200, seed=42):
        if minibatch and use_logit:
            weights_path = '/mnt/gpid08/users/albert.dominguez/weights/minibatch_kmeans_logit317K.pt'
            if os.path.exists(weights_path):
                log.info('Loading k-means cluster centroids from precomputed file.')
                return torch.load(weights_path)
        log.info('Getting k-Means cluster centroids...')
        if not isinstance(k, Iterable):
            if minibatch:
                k_means_obj = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=seed, n_init=1, max_iter=10).fit(X)
            else:
                k_means_obj = KMeans(n_clusters=k, random_state=42, n_init=1, max_iter=10).fit(X)
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
                k_means_objs = [MiniBatchKMeans(n_clusters=i, batch_size=batch_size, random_state=seed, n_init=1, max_iter=10, compute_labels=False).fit(X)
                                for i in k]
            else:
                k_means_objs = [KMeans(n_clusters=i, random_state=42, n_init=1, max_iter=10, compute_labels=False).fit(X) for i in k]
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


class KMeansInitializationTorch(object):
    @classmethod
    def get_decoder_init(cls, X, K, use_logit):
        log.info('Getting torch-based k-Means cluster centroids...')
        P_init = torch.cat([cls.kMeans_fit(X, k)[1] for k in K]).float()
        if not use_logit:
            return P_init
        P_init = torch.logit(torch.clamp(P_init, min=1e-4, max=1-1e-4), eps=1e-4)
        if sum(torch.isnan(P_init.flatten())).item() > 0:
            log.error('Initialization weights contain NaN values.')
            raise Exception
        log.info('All k-means run successfully')
        return P_init


    @classmethod
    def kMeans_fit(cls, x, K, Niter=100, verbose=True):
        """Implements Lloyd's algorithm for the Euclidean metric."""
        log.info(f'k-Means for k={K}')
        Niter = 10
        start = time.time()
        N, D = x.shape  # Number of samples, dimension of the ambient space
        x = torch.tensor(x, dtype=torch.float)
        c = x[:K, :].clone()  # Simplistic initialization for the centroids

        x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
        c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

        # K-means loop:
        # - x  is the (N, D) point cloud,
        # - cl is the (N,) vector of class labels
        # - c  is the (K, D) cloud of cluster centroids
        for i in range(Niter):

            # E step: assign points to the closest cluster -------------------------
            D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
            cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

            # M step: update the centroids to the normalized cluster average: ------
            # Compute the sum of points per cluster:
            c.zero_()
            c.scatter_add_(0, cl[:, None].repeat(1, D), x)

            # Divide by the number of points per cluster:
            Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
            c /= Ncl  # in-place division to compute the average

        if verbose:  # Fancy display -----------------------------------------------
            if use_cuda:
                torch.cuda.synchronize()
            end = time.time()
            log.info(
                f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
            )
            log.info(
                "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                    Niter, end - start, Niter, (end - start) / Niter
                )
            )
        return cl, c.cpu()


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


if __name__ == '__main__':
    import utils
    trX, _, _, _ = utils.read_data('317K')
    P_init = KMeansInitialization.get_decoder_init(trX, [3,4,5,6,7,8,9,10], minibatch=True, use_logit=True)
    log.info('Saving decoder weights initialization...')
    torch.save(P_init, '/mnt/gpid08/users/albert.dominguez/weights/minibatch_kmeans_logit317K.pt')
    print('Done!')
