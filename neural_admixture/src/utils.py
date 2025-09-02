import logging
import random
import os
import sys
import numpy as np
import torch

from pathlib import Path

from .snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

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
