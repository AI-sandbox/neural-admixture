import configargparse
import logging
import random
import os
import sys
import numpy as np
import torch

from pathlib import Path
from typing import List

from .snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def parse_train_args(argv: List[str]):
    """Training arguments parser
    """
    parser = configargparse.ArgumentParser(prog='neural-admixture train',
                                           description='Rapid population clustering with autoencoders - training mode',
                                           config_file_parser_class=configargparse.YAMLConfigFileParser)
    
    parser.add_argument('--epochs', required=False, type=int, default=250, help='Maximum number of epochs.')
    parser.add_argument('--batch_size', required=False, default=800, type=int, help='Batch size.')
    parser.add_argument('--learning_rate', required=False, default=20e-4, type=float, help='Learning rate.')

    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    parser.add_argument('--k', required=False, type=int, help='Number of populations/clusters.')
    parser.add_argument('--min_k', required=False, type=int, help='Minimum number of populations/clusters (multi-head)')
    parser.add_argument('--max_k', required=False, type=int, help='Maximum number of populations/clusters (multi-head)')
    parser.add_argument('--hidden_size', required=False, default=1024, type=int, help='Dimension of first projection in encoder.')
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name')
    
    parser.add_argument('--supervised_loss_weight', required=False, default=100, type=float, help='Weight given to the supervised loss')
    parser.add_argument('--pops_path', required=False, default='', type=str, help='Path containing the main data populations')
    
    parser.add_argument('--n_components', required=False, type=int, default=8, help='Number of components to use for the SVD initialization.')
    
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
    parser.add_argument('--num_gpus', required=False, default=0, type=int, help='Number of GPUs to be used in the execution.')
    return parser.parse_args(argv)

def read_data(tr_file: str, tr_pops_f: str=None) -> np.ndarray:
    """
    Reads SNP data from a file and applies imputation if specified..

    Args:
        tr_file (str): Path to the SNP data file.

    Returns:
        np.ndarray: A numpy array containing the SNP data.
    """
    snp_reader = SNPReader()
    data = snp_reader.read_data(tr_file)
    log.info(f"    Data contains {data.shape[0]} samples and {data.shape[1]} SNPs.")
    if tr_pops_f:
        log.info("    Population file provided!")
        with open(tr_pops_f, 'r') as fb:
            pops = [p.strip() for p in fb.readlines()]
    else:
        pops = None
    return data, pops, data.shape[0], data.shape[1]

def write_outputs(Qs: np.ndarray, run_name: str, K: int, min_k: int, max_k: int, out_path: str, Ps: np.ndarray = None) -> None:
    """
    Save the Q and optional P matrices to specified output files.

    Args:
        Qs (list of numpy.ndarray): List of Q matrices to be saved.
        run_name (str): Identifier for the run, used in file naming.
        K (int): Number of clusters, included in the file name.
        min_k (int): Minimum number of clusters (for range output).
        max_k (int): Maximum number of clusters (for range output).
        out_path (str or Path): Directory where the output files should be saved.
        Ps (list of numpy.ndarray, optional): List of P matrices to be saved, if provided.

    Returns:
        None
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if K is not None:
        np.savetxt(out_path / f"{run_name}.{K}.Q", Qs[0], delimiter=' ')
        if Ps is not None:
            np.savetxt(out_path / f"{run_name}.{K}.P", Ps[0], delimiter=' ')
            log.info("    Q and P matrices saved.")
        else:
            log.info("    Q matrix saved.")
    else:
        for i, K in enumerate(range(min_k, max_k + 1)):
            np.savetxt(out_path / f"{run_name}.{K}.Q", Qs[i], delimiter=' ')
            if Ps is not None:
                np.savetxt(out_path / f"{run_name}.{K}.P", Ps[i], delimiter=' ')
        log.info("    Q and P matrices saved for all K." if Ps is not None else "    Q matrices saved for all K.")

def ddp_setup(stage: str, rank: int, world_size: int) -> None:
    """
    Set up the distributed environment for training.

    Args:
        stage (str): Either 'begin' to initialize or 'end' to finalize the distributed process group.
        rank (int): The rank (ID) of the current process.
        world_size (int): The total number of processes participating in the training.

    Returns:
        None
    """
    if world_size > 1:
        if stage == 'begin':
            os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
            os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")

            torch.cuda.set_device(rank % torch.cuda.device_count())

            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size
            )
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
