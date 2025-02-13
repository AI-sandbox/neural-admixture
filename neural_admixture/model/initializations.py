import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
import time
import torch
import os

from pathlib import Path
from typing import List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans as KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture as GaussianMixture
from scipy.optimize import linear_sum_assignment as linear_assignment

#from models import GaussianMixture
from ..src.ipca_gpu import GPUIncrementalPCA
from .neural_admixture import NeuralAdmixture

torch.serialization.add_safe_globals([GPUIncrementalPCA])

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def determine_device_for_tensors(data_shape: tuple, K: int, device: torch.device, memory_threshold: float = 0.9) -> torch.device:
    """
    Determine if tensors can fit in GPU memory and return appropriate device.
    
    Args:
        data_shape: Shape of the input data tensor
        K: Number of components/clusters
        device: Current torch device
        memory_threshold: Fraction of available GPU memory to use (default: 0.9)
        
    Returns:
        torch.device: Device to use for tensors ('cuda' if they fit, 'cpu' if they don't)
    """
    def bytes_to_human_readable(bytes_value: int) -> str:
        """Convert bytes to human readable string (GB/MB)"""
        gb = bytes_value / (1024**3)
        if gb >= 1:
            return f"{gb:.2f} GB"
        mb = bytes_value / (1024**2)
        return f"{mb:.2f} MB"

    def calculate_tensor_memory(shape, dtype=torch.float32) -> int:
        """Calculate memory required for tensor of given shape and dtype"""
        num_elements = np.prod(shape)
        bytes_per_element = dtype.itemsize
        return num_elements * bytes_per_element
    
    if 'cuda' in device.type:
        available_gpu_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
        
        memory_required = {
            'P': calculate_tensor_memory((data_shape[1], K)),
            'Q': calculate_tensor_memory((data_shape[0], K)),
            'data': calculate_tensor_memory(data_shape, 
                                        dtype=torch.bfloat16 if 'cuda' in device.type else torch.float32)
        }
        
        total_memory_required = sum(memory_required.values())
        fits_in_gpu = total_memory_required <= (available_gpu_memory * memory_threshold)
        device_tensors = device if fits_in_gpu else 'cpu'
        
        if str(device) == 'cuda:0':
            log.info(f"    Tensors stored in {('GPU' if fits_in_gpu else 'CPU')} because there are "
            f"{bytes_to_human_readable(available_gpu_memory)} available in GPU and "
            f"tensors occupy {bytes_to_human_readable(total_memory_required)}")
    else:
        device_tensors = device
        
    return device_tensors

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
    return

def load_or_compute_pca(path: Optional[str], X: np.ndarray, n_components: int, batch_size: int, 
                        device: torch.device, run_name: str, master: bool, sample_fraction: Optional[float]=None):
    """
    Load a PCA object from a file or compute it if not found, with an option to train on a random sample of the data.

    Args:
        path: str or None, the path to the PCA object file.
        X: Numpy array, the input data to fit the PCA.
        n_components: str, the number of components for PCA.
        batch_size: int, the batch size for IncrementalPCA.
        sample_fraction: float or None, the fraction of rows to sample from X (between 0 and 1).
        master (bool): Whether or not this process is the master for printing the output.
    
    Returns:
        X_pca: numpy array, the transformed input data.
        pca_obj: the trained PCA object.
    """
    # Save original X in case of later transformation
    X_original = X
    
    try:
        if os.path.exists(path):
            pca_obj = torch.load(path, weights_only=True, map_location=device)
            assert pca_obj.n_features_in_ == X_original.shape[1], "Computed PCA and training data do not have the same number of features"
            pca_obj.to(device)
            if master:
                log.info("            PCA loaded.")
            X_pca = pca_obj.transform(X).cpu()
        else:
            raise FileNotFoundError
        
    except (FileNotFoundError, IOError, AssertionError):
        # Optionally reduce the number of rows by sampling for training
        if sample_fraction is not None and 0 < sample_fraction < 1:
            num_rows = X.shape[0]
            sampled_indices = np.random.choice(num_rows, size=int(sample_fraction * num_rows), replace=False)
            sampled_indices = np.sort(sampled_indices)
            X = X[sampled_indices, :]
            if master:
                log.info(f"    Using {int(sample_fraction * num_rows)}/{num_rows} to compute PCA.")
                
        pca_obj = GPUIncrementalPCA(n_components=int(n_components), batch_size=batch_size, device=device)
        pca_obj.fit(X)
        X_pca = pca_obj.transform(X_original).cpu()
        assert pca_obj.n_features_in_ == X_original.shape[1], "Computed PCA and training data do not have the same number of features"

        if path is not None:
            torch.save(pca_obj.cpu(), path)
        if master:
            log.info(f"            {n_components}D PCA object not found. Performing IncrementalPCA...")
            
    try:
        if path is not None:
            plot_save_path = Path(path).parent / f"{run_name}_training_pca.png"
            pca_plot(X_pca, plot_save_path)
    except Exception as e:
        log.warning(f"    Could not render PCA plot: {e}")
        log.info("    Resuming...")
    
    return X_pca, pca_obj

class GMMInitialization(object):
    """
    Class to initialize a neural admixture model using Gaussian Mixture Models (GMM).
    """
    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, init_path: Path, 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, num_gpus: int, hidden_size: int, 
                        activation: torch.nn.Module, master: bool, num_cpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
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
            log.info("    Running Gaussian Mixture initialization...")
            log.info("")
        t0 = time.time()
        X_pca, pca_obj = load_or_compute_pca(init_path, data, n_components, 1024, device, name, master, sample_fraction=1)
        te = time.time()
        if master:
            log.info(f"            PCA initialized in {te-t0:.3f} seconds.")
            log.info("")

        gmm = GaussianMixture(n_components=K, n_init=3, init_params='k-means++', tol=1e-4, covariance_type='full')
        #gmm = GaussianMixture(n_components=K, n_features=X_pca.shape[1], init_params='kmeans++', covariance_type='full', device=device).to(device)
        
        gmm.fit(X_pca.numpy())
        
        device_tensors = determine_device_for_tensors(data.shape, K, device)
        #gmm.mu.squeeze(0)
        P_init = torch.as_tensor(pca_obj.inverse_transform(gmm.means_), dtype=torch.float32, device=device_tensors).T.contiguous()
        data = torch.as_tensor(data, dtype=torch.float32, device=device_tensors)
        input = X_pca.to(device_tensors)
        
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus)
        
        P, Q, raw_model = model.launch_training(P_init, data, hidden_size, X_pca.shape[1], K, activation, input)

        return P, Q, raw_model

class KMeansInitialization(object):
    """
    Class to initialize a neural admixture model using consensus K-Means.
    """
    @classmethod
    def single_clustering_run(cls, X_pca: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
        """
        Performs a single run of K-Means clustering.

        Args:
            X_pca (np.ndarray): Input data after PCA.
            n_clusters (int): Number of clusters.
            seed (int): Random seed for reproducibility.

        Returns:
            np.ndarray: Cluster centers.
        """
        k_means_obj = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, random_state = seed)
        k_means_obj.fit(X_pca)
        centers = k_means_obj.cluster_centers_
        return centers

    @staticmethod
    def align_clusters(reference_centers: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Aligns cluster centers to a reference using the Hungarian algorithm.

        Args:
            reference_centers (np.ndarray): Reference cluster centers.
            centers (np.ndarray): Cluster centers to align.

        Returns:
            np.ndarray: Aligned cluster centers.
        """
        D = pairwise_distances(reference_centers, centers)
        _, col_ind = linear_assignment(D)
        return centers[col_ind]

    @classmethod
    def consensus_clustering(cls, X_pca: np.ndarray, n_clusters: int, seeds: List[int]) -> np.ndarray:
        """
        Computes consensus cluster centers from multiple K-Means runs.

        Args:
            X_pca (np.ndarray): Input data after PCA.
            n_clusters (int): Number of clusters.
            seeds (List[int]): List of random seeds.

        Returns:
            np.ndarray: Consensus cluster centers.
        """
        all_centers = [cls.single_clustering_run(X_pca, n_clusters, seed) for seed in seeds]
        reference_centers = all_centers[0]
        aligned_centers = [cls.align_clusters(reference_centers, centers) for centers in all_centers[1:]]
        avg_centers = np.mean([reference_centers] + aligned_centers, axis=0)
        return avg_centers

    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, init_path: Path, 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, num_gpus: int, hidden_size: int, 
                        activation: torch.nn.Module, master: bool, num_cpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
        """
        Initializes P and Q matrices and trains a neural admixture model using consensus K-Means.

        Args:
            epochs (int): Number of epochs.
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
            log.info("    Running KMeans initialization...")
            log.info("")
        t0 = time.time()
        X_pca, pca_obj = load_or_compute_pca(init_path, data, n_components, 1024, device, name, master, sample_fraction=1)
        te = time.time()
        if master:
            log.info(f'            PCA initialized in {te-t0} seconds.')
            log.info("")

        n_runs = 10 
        rng = np.random.default_rng(seed) 
        seeds = rng.integers(low=0, high=10000, size=n_runs)
        
        avg_centers = cls.consensus_clustering(X_pca, K, seeds)
        final_k_means = KMeans(n_clusters=K, init=avg_centers, n_init=1, max_iter=300, random_state=seed)
        final_k_means.fit(X_pca.numpy())
        
        device_tensors = determine_device_for_tensors(data.shape, K, device)
                    
        P_init = torch.as_tensor(pca_obj.inverse_transform(final_k_means.cluster_centers_), dtype=torch.float32, device=device_tensors).T.contiguous()
        data = torch.as_tensor(data, dtype=torch.float32, device=device_tensors)
        input = X_pca.to(device_tensors)
        
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus)
        
        P, Q, raw_model = model.launch_training(P_init, data, hidden_size, X_pca.shape[1], K, activation, input)

        return P, Q, raw_model

class RandomInitialization(object):
    """
    Class to initialize a neural admixture model using Gaussian Mixture Models (GMM).
    """
    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, init_path: Path, 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, num_gpus: int, hidden_size: int, 
                        activation: torch.nn.Module, master: bool, num_cpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
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
            log.info("")
        t0 = time.time()
        X_pca, _ = load_or_compute_pca(init_path, data, n_components, 1024, device, name, master, sample_fraction=1)
        te = time.time()
        if master:
            log.info(f"            PCA initialized in {te-t0:.3f} seconds.")
            log.info("")
        
        device_tensors = determine_device_for_tensors(data.shape, K, device)

        indices = np.random.choice(data.shape[0], K, replace=False)
        P_init = torch.as_tensor(data[indices, :], dtype=torch.float32, device=device).T.contiguous()
        data = torch.as_tensor(data, dtype=torch.float32, device=device_tensors)
        input = X_pca.to(device_tensors)
        
        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus)
        
        P, Q, raw_model = model.launch_training(P_init, data, hidden_size, X_pca.shape[1], K, activation, input)

        return P, Q, raw_model

class SupervisedInitialization(object):
    """Supervised initialization
    """
    @classmethod
    def get_decoder_init(cls, epochs: int, batch_size: int, learning_rate: float, K: int, seed: int, init_path: Path, 
                        name: str, n_components: int, data: np.ndarray, device: torch.device, num_gpus: int, hidden_size: int, 
                        activation: torch.nn.Module, master: bool, num_cpus: int, y: str, supervised_loss_weight: float) -> Tuple[torch.Tensor, torch.Tensor, torch.nn.Module]:
        if master:
            log.info("    Running Supervised initialization...")
        assert y is not None, 'Ground truth ancestries needed for supervised mode'
        t0 = time.time()
        X_pca, _ = load_or_compute_pca(init_path, data, n_components, 1024, device, name, master, sample_fraction=1)
        te = time.time()
        if master:
            log.info(f"            PCA initialized in {te-t0} seconds.")

        ancestry_dict = {anc: idx for idx, anc in enumerate(sorted(np.unique([a for a in y if a != '-'])))}
        assert len(ancestry_dict) == K, f'Number of ancestries in training ground truth ({len(ancestry_dict)}) is not equal to the value of K ({K})'
        ancestry_dict['-'] = -1
        to_idx_mapper = np.vectorize(lambda x: ancestry_dict[x])
        
        # Do not take into account samples with missing labels
        y_num = to_idx_mapper(y[:])
        mask = y_num > -1
        masked_y_num = y_num[mask]
        X_masked = data[mask,:]
        
        device_tensors = determine_device_for_tensors(data.shape, K, device)
        
        P_init = torch.as_tensor(np.vstack([X_masked[masked_y_num==idx,:].mean(axis=0) for idx in range(K)]), dtype=torch.float32, device=device_tensors).T.contiguous()
        data = torch.as_tensor(data, dtype=torch.bfloat16 if 'cuda' in device.type else torch.float32, device=device_tensors)
        input = X_pca.to(device_tensors)
        y = torch.as_tensor(y_num, dtype=torch.int64, device=device_tensors)

        model = NeuralAdmixture(K, epochs, batch_size, learning_rate, device, seed, num_gpus, device_tensors, master, num_cpus, supervised_loss_weight)
        
        P, Q, raw_model = model.launch_training(P_init, data, hidden_size, X_pca.shape[1], K, activation, input, y=y)
       
        return P, Q, raw_model