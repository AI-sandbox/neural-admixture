import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import time
import torch
from collections.abc import Iterable
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.decomposition import PCA, TruncatedSVD


logging.basicConfig(level=logging.INFO)
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

class KMeansInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, k, minibatch, use_logit, batch_size=200, seed=42, path=''):
        if minibatch and use_logit:
            if os.path.exists(path):
                log.info('Loading k-means cluster centroids from precomputed tensor.')
                return torch.load(path)
        log.info('Computing k-Means cluster centroids...')
        if not isinstance(k, Iterable):
            if minibatch:
                k_means_obj = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, random_state=seed, n_init=3, max_iter=3).fit(X)
            else:
                k_means_obj = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=3).fit(X)
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
                k_means_objs = [MiniBatchKMeans(n_clusters=i, batch_size=batch_size, random_state=seed, n_init=3, max_iter=3, compute_labels=False).fit(X)
                                for i in k]
            else:
                k_means_objs = [KMeans(n_clusters=i, random_state=42, n_init=3, max_iter=3).fit(X) for i in k]
            if not use_logit:
                return torch.cat([torch.tensor(obj.cluster_centers_) for obj in k_means_objs], axis=0).float()
            P_init = torch.clamp(torch.cat([torch.tensor(obj.cluster_centers_) for obj in k_means_objs], axis=0).float(),
                                 min=1e-4, max=1-1e-4)
            del k_means_objs
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                raise Exception
            torch.save(P_init, path)
            log.info('All k-means run successfully. Initialization was saved.')
            return P_init


class KMeansPlusPlusInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, k, seed=42, path=''):
        if os.path.exists(path):
            log.info('Loading k-means++ centroids from precomputed tensor.')
            return torch.load(path)
        log.info('Computing k-Means++ centroids...')
        X_np = np.array(X[:])
        if not isinstance(k, Iterable):
            centers, _ = kmeans_plusplus(X_np, n_clusters=k, random_state=seed)
            P_init = torch.clamp(torch.tensor(centers).float(), min=1e-4, max=1-1e-4) 
            del centers, X_np
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
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                raise Exception
            torch.save(P_init, path)
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
    def get_decoder_init(cls, X, K, path=''):
        if os.path.exists(path):
            log.info('Loading binomial initialization from precomputed tensor.')
            return torch.load(path)
        log.info(f'Running binomial initialization...')
        np.random.seed(42)
        P_inits = [torch.tensor(np.random.binomial(n=1, p=0.1, size=X.shape[1]), dtype=torch.float) for _ in range(sum(K))]
        P_init = torch.logit(torch.vstack(P_inits), eps=1e-4)
        del P_inits
        torch.save(P_init, path)
        return P_init

class PCAInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, K, path=''):
        if len(K) > 1:
            raise NotImplementedError
        k = K[0]
        if k <= 4 or k > 9:
            raise NotImplementedError
        log.info('Running PCA initialization...')
        try:
            with open(path, 'rb') as fb:
                pca_obj = pickle.load(fb)
            log.info('PCA loaded.') 
        except FileNotFoundError as fnfe:
            log.info('PCA object not found. Performing PCA...')
            pca_obj = PCA(n_components=2, random_state=42)
            pca_obj.fit(X)
            with open(path, 'wb') as fb:
                pickle.dump(pca_obj, fb)
        except Exception as e:
            raise e
        X_pca = pca_obj.transform(X)
        svs = np.sqrt(pca_obj.singular_values_)
        part1 = np.array([-svs[0]/2,0,svs[0]/2])
        part2 = np.array([-svs[1]/2,0,svs[1]/2])
        xv, yv = np.meshgrid(part1, part2)
        concat = np.concatenate((xv.reshape(-1,1), yv.reshape(-1,1)), axis=1)
        init_points = [None]*len(concat)
        for i, p in enumerate(concat):
            rep = np.vstack([p]*X.shape[0])
            init_points[i] = X_pca[np.argmin(np.linalg.norm(rep-X_pca, axis=1))]
        init_points = np.vstack(init_points)
        np.random.seed(42)
        inv_tr = pca_obj.inverse_transform(init_points)
        idxs = np.random.choice(inv_tr.shape[0], k, replace=False)
        P_init = torch.tensor(inv_tr[idxs,:], dtype=torch.float)
        log.info('Weights initialized.')
        return P_init

class AdmixtureInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, K, path):
        log.info('Fetching ADMIXTURE weights...')
        if len(K) > 1:
            raise NotImplementedError
        # Loads standard ADMIXTURE output format
        P_init = torch.tensor(1-np.genfromtxt(path, delimiter=' ').T, dtype=torch.float32)
        assert P_init.size()[0] == K[0], 'Input P is not coherent with the value of K'
        log.info('Weights fetched.')
        return P_init

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
            P_init = torch.tensor(pca_obj.inverse_transform(centers), dtype=torch.float32).view(sum(K), -1)
        else:
            k_means_obj = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=10).fit(X_tsvd)
            P_init = torch.tensor(pca_obj.inverse_transform(k_means_obj.cluster_centers_), dtype=torch.float32).view(K, -1)
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


class TSVDKMeansInitialization(object):
    @classmethod
    def get_decoder_init(cls, X, K, path, run_name, n_components):
        log.info('Running TSVD-KMeans initialization...')
        t0 = time.time()
        try:
            with open(path, 'rb') as fb:
                tsvd_obj = pickle.load(fb)
            log.info('TSVD loaded.')
            if tsvd_obj.components_.shape[0] != n_components:
                raise FileNotFoundError
        except FileNotFoundError as fnfe:
            log.info(f'{n_components}D TSVD object not found. Performing TSVD...')
            tsvd_obj = TruncatedSVD(n_components=n_components, random_state=42)
            tsvd_obj.fit(X)
            with open(path, 'wb') as fb:
                pickle.dump(tsvd_obj, fb)
        except Exception as e:
            raise e
        assert tsvd_obj.components_.shape[1] == X.shape[1], 'Computed TSVD and training data do not have same number of SNPs' 
        log.info('Projecting data...')
        X_tsvd = tsvd_obj.transform(X)
        log.info('Running KMeans on projected data...')
        if isinstance(K, Iterable):
            k_means_objs = [KMeans(n_clusters=i, random_state=42, n_init=10, max_iter=10).fit(X_tsvd) for i in K]
            centers = np.concatenate([obj.cluster_centers_ for obj in k_means_objs])
            P_init = torch.tensor(tsvd_obj.inverse_transform(centers), dtype=torch.float32).view(sum(K), -1)
        else:
            k_means_obj = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=10).fit(X_tsvd)
            P_init = torch.tensor(tsvd_obj.inverse_transform(k_means_obj.cluster_centers_), dtype=torch.float32).view(K, -1)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        log.info('Rendering TSVD plot...')
        try:
            save_root = '/'.join(path.split('/')[:-1]) if len(path.split('/')) > 1 else '.'
            plot_save_path = f'{save_root}/{run_name}_training_tsvd.png'
            pca_plot(X_tsvd, plot_save_path)
        except Exception as e:
            log.warn(f'Could not render TSVD plot: {e}')
            log.info('Resuming...')
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
        assert len(ancestry_dict) == k, 'Number of ancestries in training ground truth is not equal to the value of k'
        to_idx_mapper = np.vectorize(lambda x: ancestry_dict[x])
        y_num = to_idx_mapper(y[:])
        X_mem = X[:,:]
        P_init = torch.tensor(np.vstack([X_mem[y_num==idx,:].mean(axis=0) for idx in range(k)]), dtype=torch.float32)
        te = time.time()
        log.info('Weights initialized in {} seconds.'.format(te-t0))
        return P_init