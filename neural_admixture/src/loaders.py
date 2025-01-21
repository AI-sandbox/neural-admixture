import torch
import math

from torch.utils.data import Dataset, DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple, Union, List, Generator

# DATALOADERS:
def dataloader_P1(data: torch.Tensor, Q: torch.Tensor, batch_size: int, num_gpus: int, seed: int, 
                generator: torch.Generator, pin: bool, num_cpus: int) -> Tuple[DataLoader, Union[BatchSampler, DistributedSampler]]:
    """
    Creates a DataLoader with batch sampler or distributed sampler for the phase 1.

    Parameters:
    - data (torch.Tensor): Input tensor data.
    - Q (torch.Tensor): Tensor associated with Q values.
    - batch_size (int): Size of each batch.
    - num_gpus (int): Number of GPUs available for distributed training.
    - seed (int): Seed for random number generation.
    - generator (torch.Generator): Random number generator instance.
    - pin (bool): Whether to pin memory for data loading.

    Returns:
    - Tuple[DataLoader, Union[BatchSampler, DistributedSampler]]: DataLoader object and the used sampler.
    """
    dataset = Dataset_P1(data, Q)
    if num_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn_P1, pin_memory=pin)
    elif num_gpus == 1:
        sampler = BatchSampler(dataset, batch_size, generator, shuffle=True, seed=seed)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn_P1, pin_memory=pin)
    else:
        sampler = BatchSampler(dataset, batch_size, generator, shuffle=True, seed=seed)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn_P1, num_workers=max(2, min(num_cpus - 1, int(num_cpus * 0.065))))
    return dataloader, sampler

def dataloader_P2(X: torch.Tensor, input: torch.Tensor, batch_size: int, num_gpus: int, seed: int, 
                generator: torch.Generator, pin: bool, y: torch.Tensor, num_cpus: int) -> Tuple[DataLoader, Union[BatchSampler, DistributedSampler]]:
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
    dataset = Dataset_P2(X, input, y)
    if num_gpus > 1:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn_P2, pin_memory=pin)
    elif num_gpus == 1:
        sampler = BatchSampler(dataset, batch_size, generator, seed, shuffle=True, pad=True)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn_P2, pin_memory=pin)
    else:
        sampler = BatchSampler(dataset, batch_size, generator, seed, shuffle=True, pad=True)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn_P2, num_workers=max(2, min(num_cpus - 1, int(num_cpus * 0.065))))
    return dataloader, sampler

def dataloader_inference(input: torch.Tensor, batch_size: int, seed: int, generator: torch.Generator, num_gpus: int, 
                        pin: bool, num_cpus: int) -> DataLoader:
    """
    Creates a DataLoader for inference using a BatchSampler.

    Parameters:
    - input (torch.Tensor): Input tensor used for inference.
    - batch_size (int): Size of each batch.
    - seed (int): Seed for random number generation.
    - generator (torch.Generator): Random number generator instance.

    Returns:
    - DataLoader: DataLoader object for inference.
    """
    dataset = Dataset_inference(input)
    if num_gpus == 1:
        sampler = BatchSampler(dataset, batch_size, generator, seed, shuffle=False, pad=False)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn_inference, pin_memory=pin)
    else:
        sampler = BatchSampler(dataset, batch_size, generator, seed, shuffle=False, pad=False)
        dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn_inference, num_workers=max(2, min(num_cpus - 1, int(num_cpus * 0.065))))
    return dataloader
    
# F1 DATASET:
class Dataset_P1(Dataset):
    """
    Dataset for phase 1 of Neural Admixture.
    
    Args:
        data (torch.Tensor): Tensor containing the main data of the dataset.
        Q (torch.Tensor): Tensor containing additional information associated with the data.
    """
    def __init__(self, data: torch.Tensor, Q: torch.Tensor):
        """
        Args:
            data (torch.Tensor): Tensor containing the main data of the dataset.
            Q (torch.Tensor): Tensor containing additional information associated with the data.
        """
        self.data = data
        self.Q = Q
        self.num_total_elements = data.size(0)

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of elements in the dataset.
        """
        return self.num_total_elements

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing a `q_item` tensor 
            and a `data_item` tensor corresponding to the given index.
        """
        data_item = self.data[idx]
        q_item = self.Q[idx]
        return q_item, data_item
    
def collate_fn_P1(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines a list of pairs (q_tensor, target) into stacked tensors.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples 
        containing `q_tensor` and `target` tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two stacked tensors. The first 
        contains the `q_tensors` and the second contains the `targets`.
    """
    q_tensors, targets = zip(*batch)
    q_tensors = torch.stack(q_tensors, dim=0)
    targets = torch.stack(targets, dim=0)
    return q_tensors, targets


# F2 DATASET:
class Dataset_P2(Dataset):
    """
    Dataset for phase 2 of Neural Admixture.
    
    Args:
            X (torch.Tensor): The main data tensor.
            input (torch.Tensor): The input tensor associated with the data.
    """
    def __init__(self, X: torch.Tensor, input: torch.Tensor, y: torch.Tensor):
        """
        Args:
            X (torch.Tensor): The main data tensor.
            input (torch.Tensor): The input tensor associated with the data.
        """
        self.X = X
        self.input= input
        self.y = y
        
    def __len__(self) -> int:
        """
        Returns:
            int: The number of elements in the dataset.
        """
        return self.input.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing `batch_X` and `batch_input` tensors.
        """
        batch_X = self.X[idx]
        batch_input = self.input[idx]
        batch_y = self.y[idx]
        return batch_X, batch_input, batch_y

def collate_fn_P2(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines a list of pairs (batch_X, batch_input) into stacked tensors.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): A list of tuples containing `batch_X` and `batch_input`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two stacked tensors. The first contains the `batch_X` tensors, 
        and the second contains the `batch_input` tensors.
    """
    batch_X, batch_input, batch_y = zip(*batch)
    batch_X = torch.stack(batch_X)
    batch_input = torch.stack(batch_input)
    batch_y = torch.stack(batch_y)
    return batch_X, batch_input, batch_y


# INFERENCE DATASET:
class Dataset_inference(Dataset):
    """
    Dataset for inference of Neural Admixture.
    
    Args:
        input (torch.Tensor): The input tensor associated with the data.
    """
    def __init__(self, input: torch.Tensor):
        """
        Args:
            input (torch.Tensor): The input tensor associated with the data.
        """
        self.input= input
        
    def __len__(self) -> int:
        """
        Returns:
            int: The number of elements in the dataset.
        """
        return self.input.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            torch.Tensor: The input tensor corresponding to the given index.
        """
        batch_input = self.input[idx]
        return batch_input

def collate_fn_inference(batch: List[torch.Tensor]) -> torch.Tensor:
    """
    Combines a list of input tensors into a single stacked tensor.

    Args:
        batch (List[torch.Tensor]): A list of input tensors.

    Returns:
        torch.Tensor: A stacked tensor containing all the inputs in the batch.
    """
    batch_input = torch.stack(batch)
    return batch_input


# BATCH SAMPLER:
class BatchSampler(BatchSampler):
    """
    Batch Sampler of Neural Admixture used when the number of GPUs is lower or equal to 1.
    
    Args:
        dataset (Dataset): The dataset from which to sample.
        batch_size (int): The size of each batch.
        generator (torch.Generator): The random number generator.
        seed (int, optional): The seed for random sampling. Default is 0.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
        pad (bool, optional): Whether to pad the data to a multiple of the batch size. Default is True.
    """
    def __init__(self, dataset: Dataset, batch_size: int, generator: torch.Generator, seed: int = 0, 
                 shuffle: bool = True, pad: bool = True):
        """
        Args:
            dataset (Dataset): The dataset from which to sample.
            batch_size (int): The size of each batch.
            generator (torch.Generator): The random number generator.
            seed (int, optional): The seed for random sampling. Default is 0.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.
            pad (bool, optional): Whether to pad the data to a multiple of the batch size. Default is True.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(self.dataset)
        self.total_size = self.num_samples
        self.num_batches = math.ceil(self.num_samples / self.batch_size)
        padding_size = self.num_batches * self.batch_size - self.num_samples
        self.total_size = self.num_samples + padding_size
        self.generator = generator
        self.epoch = 0
        self.pad = pad

    def __iter__(self) -> Generator[list[int], None, None]:
        """
        Returns:
            Generator[list[int], None, None]: A generator that yields batches of indices.
        """
        if self.shuffle:
            self.generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.pad:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

            assert len(indices) == self.total_size

        # Yield batches
        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            yield indices[start_idx:end_idx]

    def __len__(self) -> int:
        """
        Returns:
            int: The number of batches in the sampler.
        """
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch number for the sampler.

        Args:
            epoch (int): The current epoch number.
        """
        self.epoch = epoch