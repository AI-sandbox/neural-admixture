"""
Adapted from ... # TODO: add source
"""
import torch
from typing import Optional, Tuple, Union
import numpy as np

class GPUIncrementalPCA:
    """
    Incremental Principal Component Analysis with GPU Acceleration.
    
    This implementation processes data in batches and can utilize GPU acceleration
    for faster computation on large datasets.
    
    Parameters:
        n_components: Number of components to keep. If None, keeps all components.
        whiten: When True, components_ vectors are divided by singular values to ensure
                uncorrelated outputs with unit component-wise variances.
        copy: If False, X will be overwritten whenever possible.
        batch_size: The number of samples to use for each batch.
        use_fp16: If True, use half precision (float16) on GPU.
        device: Device to use for computations ('cpu', 'cuda', 'cuda:0', etc.).
    
    Attributes:
        components_: Principal axes in feature space.
        explained_variance_: Amount of variance explained by each component.
        explained_variance_ratio_: Ratio of variance explained to total variance.
        singular_values_: The singular values corresponding to each component.
        mean_: Per-feature empirical mean.
        n_components_: The estimated number of components.
        n_samples_seen_: The number of samples processed.
        noise_variance_: The estimated noise covariance.
    """
    
    def __init__(self, n_components: Optional[int] = None, *, whiten: bool = False, 
                copy: bool = True, batch_size: Optional[int] = None, use_fp64: bool = False,
                device: Optional[Union[str, torch.device]] = torch.device('cpu')):
        
        # INITILIZE MODEL CHARACTERISTICS:
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        
        # INITIALIZE MODEL SETUP:
        self.device = device
        self.dtype = torch.float64 if use_fp64 and self.device.type.startswith('cuda') else torch.float32
        
        # INITIALIZE MODEL ATRIBUTES:
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None
        self.n_components_ = None
        
    def _validate_data(self, X: Union[np.ndarray, torch.Tensor], 
                      copy: bool = True) -> torch.Tensor:
        """
        Validate and convert input data to tensor with proper dtype and device.
        
        Args:
            X: Input data
            copy: Whether to copy the data
            
        Returns:
            Validated tensor on the correct device with correct dtype
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
            
        if X.dtype != self.dtype or X.device != self.device:
            X = X.to(dtype=self.dtype, device=self.device)
        elif copy:
            X = X.clone()
            
        return X
    
    def _check_memory(self, X: torch.Tensor) -> bool:
        """
        Check if there's enough GPU memory for the operation.
        
        Args:
            X: Input tensor to check memory requirements for
            
        Returns:
            bool: True if enough memory is available, False otherwise
        """
        if self.device.type.startswith('cuda'):
            device_idx = 0 if self.device.index is None else self.device.index
            required_memory = X.element_size() * X.nelement() * 3
            available_memory = (torch.cuda.get_device_properties(device_idx).total_memory - 
                              torch.cuda.memory_allocated(device_idx))
            return required_memory < available_memory
        return True

    def to(self, device: Union[str, torch.device]) -> 'GPUIncrementalPCA':
        """
        Move the model to a different device.
        
        Args:
            device: The target device
            
        Returns:
            self: The model instance on the new device
        """
        new_device = torch.device(device)
        
        if new_device.type.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is not available.")
        
        # Move all tensors to the new device
        for attr in ['mean_', 'var_', 'components_', 'singular_values_', 
                    'explained_variance_', 'explained_variance_ratio_']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(new_device))
            
        self.device = new_device
        return self
    
    def cpu(self) -> 'GPUIncrementalPCA':
        """
        Move the model to CPU device.
        
        Returns:
            self: The model instance moved to CPU
        """
        return self.to(torch.device('cpu'))
    
    @staticmethod
    def _incremental_mean_and_var(X: torch.Tensor, 
                                 last_mean: Optional[torch.Tensor], 
                                 last_variance: Optional[torch.Tensor], 
                                 last_sample_count: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Compute incremental mean and variance.
        """
        if X.shape[0] == 0:
            return last_mean, last_variance, last_sample_count.item()

        # Initialize if first batch
        if last_mean is None:
            last_mean = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)
        if last_variance is None:
            last_variance = torch.zeros(X.shape[1], device=X.device, dtype=X.dtype)

        # Current batch statistics
        new_sample_count = X.shape[0]
        new_mean = torch.mean(X, dim=0)
        new_sum_square = torch.sum((X - new_mean.unsqueeze(0)) ** 2, dim=0)
        
        # Update counts
        updated_sample_count = last_sample_count.item() + new_sample_count
        
        # Update mean and variance using Welford's online algorithm
        delta = new_mean - last_mean
        updated_mean = last_mean + delta * (new_sample_count / updated_sample_count)
        
        m_a = last_variance * last_sample_count.item()
        m_b = new_sum_square
        m2 = m_a + m_b + delta ** 2 * last_sample_count.item() * new_sample_count / updated_sample_count
        updated_variance = m2 / updated_sample_count
        
        return updated_mean, updated_variance, updated_sample_count

    @staticmethod
    def _svd_flip(u: torch.Tensor, v: torch.Tensor, 
                  u_based_decision: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adjust signs of SVD vectors for consistency.
        """
        if u_based_decision:
            max_abs_cols = torch.argmax(torch.abs(u), dim=0)
            signs = torch.sign(u[max_abs_cols, torch.arange(u.shape[1], device=u.device)])
        else:
            max_abs_rows = torch.argmax(torch.abs(v), dim=1)
            signs = torch.sign(v[torch.arange(v.shape[0], device=v.device), max_abs_rows])
        
        u = u * signs
        v = v * signs.unsqueeze(1)
        return u, v

    @torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def fit(self, X: Union[np.ndarray, torch.Tensor], 
            check_input: bool = True) -> 'GPUIncrementalPCA':
        """
        Fit the model with X using minibatches.
        """
        if check_input:
            X = self._validate_data(X)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        # Initialize or validate n_components
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)
        else:
            self.n_components_ = min(self.n_components, n_samples, n_features)
            
        # Set batch size if not already set
        if self.batch_size is None:
            self.batch_size_ = min(5 * n_features, n_samples)
        else:
            self.batch_size_ = min(self.batch_size, n_samples)
            
        # Process data in batches
        try:
            for start in range(0, n_samples, self.batch_size_):
                end = min(start + self.batch_size_, n_samples)
                batch = X[start:end]
                
                self.partial_fit(batch, check_input=False)
                
                # Clean up GPU memory after each batch if needed
                if self.device.type.startswith('cuda'):
                    torch.cuda.empty_cache()
                    
        except RuntimeError as e:
            if "out of memory" in str(e) and self.device.type.startswith('cuda'):
                torch.cuda.empty_cache()
                self._cleanup_gpu_memory()
                
                # Reduce batch size and try again
                self.batch_size_ = self.batch_size_ // 2
                if self.batch_size_ < 1:
                    raise RuntimeError("Unable to process even with minimum batch size")
                    
                return self.fit(X, check_input=False)
            else:
                raise e
                
        return self
        
    @torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def partial_fit(self, X: torch.Tensor, check_input: bool = True) -> 'GPUIncrementalPCA':
        """
        Incremental fit with X.
        """
        if check_input:
            X = self._validate_data(X)
            
        first_pass = not hasattr(self, "components_")
        n_samples, n_features = X.shape
        
        if first_pass:
            self.components_ = None
            if self.n_components is None:
                self.n_components_ = min(n_samples, n_features)
                
        col_mean, col_var, n_total_samples = self._incremental_mean_and_var(
            X,
            self.mean_,
            self.var_,
            torch.tensor([self.n_samples_seen_], device=self.device, dtype=X.dtype)
        )
        
        # Center data
        X = X - col_mean
        
        # Update SVD with new data
        if self.n_samples_seen_ == 0:
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        else:
            mean_correction = torch.sqrt(
                torch.tensor(
                    (self.n_samples_seen_ * n_samples) / n_total_samples,
                    device=self.device,
                    dtype=X.dtype
                )
            ) * (self.mean_ - col_mean)
            
            X_combined = torch.cat([
                self.singular_values_.view(-1, 1) * self.components_,
                X,
                mean_correction.unsqueeze(0)
            ])
            
            U, S, Vt = torch.linalg.svd(X_combined, full_matrices=False)
            
        U, Vt = self._svd_flip(U, Vt)
        
        explained_variance = (S ** 2) / (n_total_samples - 1)
        explained_variance_ratio = (S ** 2) / (torch.sum(col_var * n_total_samples))
        
        self.n_samples_seen_ = n_total_samples
        self.components_ = Vt[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
        
        if self.n_components_ < min(n_features, n_samples):
            self.noise_variance_ = explained_variance[self.n_components_:].mean().item()
        else:
            self.noise_variance_ = 0.
            
        return self

    def transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Data to transform
            
        Returns:
            X_transformed: Transformed data
        """
        X = self._validate_data(X)
        
        if not hasattr(self, 'components_'):
            raise AttributeError("Model not fitted yet.")
            
        return self._transform_batch(X)

    def _transform_batch(self, X: torch.Tensor) -> torch.Tensor:
        """Internal method to transform a batch of data."""
        X_transformed = X - self.mean_
        if self.whiten:
            return torch.mm(X_transformed, 
                          (self.components_.T / self.singular_values_.view(-1, 1)))
        return torch.mm(X_transformed, self.components_.T)

    @torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def fit_transform(self, X: Union[np.ndarray, torch.Tensor], 
                     check_input: bool = True) -> torch.Tensor:
        """
        Fit the model with X and apply dimensionality reduction to X.
        """
        X = self._validate_data(X) if check_input else X
        n_samples, n_features = X.shape
        
        if self.batch_size is None:
            self.batch_size_ = min(5 * n_features, n_samples)
        else:
            self.batch_size_ = self.batch_size
            
        if not self._check_memory(X):
            transformed_data = []
            for start in range(0, n_samples, self.batch_size_):
                end = min(start + self.batch_size_, n_samples)
                X_batch = X[start:end]
                
                if start == 0:
                    batch_transformed = self.partial_fit(X_batch, check_input=False)._transform_batch(X_batch)
                else:
                    batch_transformed = self._transform_batch(X_batch)
                    
                transformed_data.append(batch_transformed)
                torch.cuda.empty_cache() if self.device.type.startswith('cuda') else None
                
            return torch.cat(transformed_data, dim=0)
        else:
            self.fit(X, check_input=False)
            return self._transform_batch(X)

    @torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    def inverse_transform(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Transform data back to its original space."""
        X = self._validate_data(X)
        
        if not hasattr(self, 'components_'):
            raise AttributeError("Model not fitted yet.")
            
        if self.whiten:
            X_transformed = torch.mm(X, 
                                   (self.components_ * self.singular_values_.view(-1, 1)))
        else:
            X_transformed = torch.mm(X, self.components_)
            
        return X_transformed + self.mean_

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory by moving unused tensors to CPU."""
        if self.device.type.startswith('cuda'):
            torch.cuda.empty_cache()
            
    def __del__(self):
        """Cleanup when object is deleted."""
        self._cleanup_gpu_memory()
