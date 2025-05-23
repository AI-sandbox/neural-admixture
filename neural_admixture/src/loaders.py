import torch

from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple

# DATALOADER:
def dataloader_admixture(X: torch.Tensor, batch_size: int, num_gpus: int, seed: int, 
                generator: torch.Generator, pops: torch.Tensor, shuffle: bool):
    """
    Creates a DataLoader with batch sampler or distributed sampler for the phase 2.

    Parameters:
    - X (torch.Tensor): Input tensor X used in the `Dataset_f2`.
    - input (torch.Tensor): Additional input tensor used in the dataset.
    - batch_size (int): Size of each batch.
    - num_gpus (int): Number of GPUs available for distributed training.
    - seed (int): Seed for random number generation.
    - generator (torch.Generator): Random number generator instance.
    - pin (bool): Whether to pin memory for data loading.

    Returns:
    - Tuple[DataLoader, Union[BatchSampler, DistributedSampler]]: DataLoader object and the used sampler.
    """
    dataset = Dataset_admixture(X, pops)
    if num_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    else:
        if shuffle:
            sampler = RandomSampler(dataset, generator=generator)
        else:
            sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return loader

# P2 DATASET:
class Dataset_admixture(Dataset):
    """
    Dataset for phase 2 of Neural Admixture.
    
    Args:
            X (torch.Tensor): The main data tensor.
            input (torch.Tensor): The input tensor associated with the data.
    """
    def __init__(self, X: torch.Tensor, pops: torch.Tensor):
        """
        Args:
            X (torch.Tensor): The main data tensor.
            input (torch.Tensor): The input tensor associated with the data.
        """
        self.X = X
        self.pops = pops
        
    def __len__(self) -> int:
        """
        Returns:
            int: The number of elements in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing `batch_X` and `batch_input` tensors.
        """
        batch_X = self.X[idx]
        batch_pops = self.pops[idx]
        return batch_X, batch_pops
