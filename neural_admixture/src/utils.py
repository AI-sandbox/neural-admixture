import configargparse
import logging
import random
import os
import sys
import socket
import numpy as np
import torch

from pathlib import Path
from typing import List, Tuple

from .snp_reader import SNPReader
from ..model.switchers import Switchers

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def parse_train_args(argv: List[str]):
    """Training arguments parser
    """
    parser = configargparse.ArgumentParser(prog='neural-admixture train',
                                           description='Rapid population clustering with autoencoders - training mode',
                                           config_file_parser_class=configargparse.YAMLConfigFileParser)
    
    parser.add_argument('--epochs', required=False, type=int, default=25, help='Maximum number of epochs.')
    parser.add_argument('--batch_size', required=False, default=800, type=int, help='Batch size.')
    parser.add_argument('--learning_rate', required=False, default=25e-4, type=float, help='Learning rate.')

    parser.add_argument('--initialization', required=False, type=str, default = 'random',
                        choices=['random'], help='P initialization.')
    
    parser.add_argument('--activation', required=False, default='relu', type=str, choices=['relu', 'tanh', 'gelu'], help='Activation function for encoder layers.')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    parser.add_argument('--k', required=True, type=int, help='Number of populations/clusters.')
    parser.add_argument('--hidden_size', required=False, default=1024, type=int, help='Dimension of first projection in encoder.')
    parser.add_argument('--n_components', required=False, type=int, default=8, help='Number of components to use for the random projection.')
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name')
    
    parser.add_argument('--num_gpus', required=False, default=0, type=int, help='Number of GPUs to be used in the execution.')
    parser.add_argument('--num_cpus', required=False, default=1, type=int, help='Number of CPUs to be used in the execution.')
    
    #parser.add_argument('--cv', required=False, default=None, type=int, help='Number of folds for cross-validation')
    return parser.parse_args(argv)

def parse_infer_args(argv: List[str]):
    """Inference arguments parser
    """
    parser = configargparse.ArgumentParser(prog='neural-admixture infer',
                                     description='Rapid population clustering with autoencoders - inference mode',
                                     config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--out_name', required=True, type=str, help='Name used to output files on inference mode.')
    parser.add_argument('--save_dir', required=True, type=str, help='Load model from this directory.')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data.')
    parser.add_argument('--name', required=True, type=str, help='Trained experiment/model name.')
    parser.add_argument('--batch_size', required=False, default=1000, type=int, help='Batch size.')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    
    parser.add_argument('--num_cpus', required=False, default=1, type=int, help='Number of CPUs to be used in the execution.')

    return parser.parse_args(argv)

def read_data(tr_file: str, master: bool) -> np.ndarray:
    """
    Reads SNP data from a file and applies imputation if specified..

    Args:
        tr_file (str): Path to the SNP data file.
        imputation (str): Type of imputation to apply ('mean' or 'zero').
        master (bool): Wheter or not this process is the master for printing the output.
        tr_pops_f (str, optional): denotes the path containing the main populations file. Defaults to None.

    Returns:
        da.core.Array: A Dask array containing the SNP data.
    """
    snp_reader = SNPReader()
    data, has_missing = snp_reader.read_data(tr_file, master)
    if master:
        log.info(f"    Data contains {data.shape[0]} samples and {data.shape[1]} SNPs.")
   
    return data, has_missing

def train(initialization: str, device: torch.device, k: int, seed: int, n_components: int, epochs: int, batch_size: int, learning_rate: float, 
        data: np.ndarray, num_gpus: int, activation_str: str, hidden_size: int, master: bool, num_cpus: int,
        has_missing: bool) -> Tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    """
    Train the model using specified initialization, hyperparameters, and data.

    Args:
        initialization (str): Initialization strategy to use.
        device (torch.device): Device to perform training (CPU or GPU).
        save_dir (str): Directory to save model initialization files.
        name (str): Name identifier for the training run.
        k (int): Number of clusters or components.
        seed (int): Random seed for reproducibility.
        n_components (int): Number of components to use in the model.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        data (np.ndarray): Training data in numpy array format.
        num_gpus (int): Number of GPUs to use.
        activation_str (str): String key for selecting the activation function.
        hidden_size (int): Size of hidden layers in the neural network.
        master (bool): Wheter or not this process is the master for printing the output.

    Returns:
        Tuple[np.ndarray, np.ndarray, torch.nn.Module]: Trained P and Q matrices and the trained model.
    """
    switchers = Switchers.get_switchers()
    activation = switchers['activations'][activation_str](0)
    P, Q = switchers['initializations'][initialization](
        epochs, batch_size, learning_rate, k, seed, n_components, data, device, 
        num_gpus, hidden_size, activation, master, num_cpus, has_missing)
    
    return P, Q

def write_outputs(Q: np.ndarray, run_name: str, K: int, out_path: str, P: np.ndarray=None) -> None:
    """
    Save the Q and optional P matrices to specified output files.

    Args:
        Q (numpy.ndarray): Q matrix to be saved.
        run_name (str): Identifier for the run, used in file naming.
        K (int): Number of clusters, included in the file name.
        out_path (str or Path): Directory where the output files should be saved.
        P (numpy.ndarray, optional): P matrix to be saved, if provided. Defaults to None.

    Returns:
        None
    """
    out_path = Path(out_path)
    np.savetxt(out_path/f"{run_name}.{K}.Q", Q, delimiter=' ')
    if P is not None:
        np.savetxt(out_path/f"{run_name}.{K}.P", P, delimiter=' ')
        log.info("    Q and P matrices saved.")
    else:
        log.info("    Q matrix saved.")
    return 

def find_free_port(start_port=12355):
    """ Find a free port starting from a given port number. """
    port = start_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
            port += 1

def ddp_setup(stage: str, rank: int, num_gpus: int) -> None:
    """
    Initialize or destroy the Distributed Data Parallel (DDP) process group.

    Args:
        stage (str): The stage of setup. 'begin' initializes the process group, 
                     while any other value destroys it. Defaults to 'begin'.

    Returns:
        None
    """
    if num_gpus>1:
        if stage == 'begin':
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(find_free_port(12355))
            torch.cuda.set_device(rank)
            torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=num_gpus, device_id=torch.device(f'cuda:{int(rank)}'))
        else:
            torch.distributed.destroy_process_group()
    
def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): Seed value.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
