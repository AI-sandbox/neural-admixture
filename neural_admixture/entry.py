import logging
import sys
import os
import torch
import torch.multiprocessing as mp
import platform
import time
import configargparse

from typing import List
from colorama import init, Fore, Style
from ._version import __version__

from .src import utils
from .src.svd import RSVD

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
    parser.add_argument('--threads', required=True, default=1, type=int, help='Number of threads to be used during execution.')
    
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
    parser.add_argument('--batch_size', required=False, default=1024, type=int, help='Batch size.')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    
    parser.add_argument('--num_gpus', required=False, default=0, type=int, help='Number of GPUs to be used in the execution.')
    parser.add_argument('--threads', required=True, default=1, type=int, help='Number of threads to be used during execution.')

    return parser.parse_args(argv)


def print_neural_admixture_banner(version: str="2.0") -> None:
    """
    Display the Neural Admixture banner with version and author information in color.
    """
    init(autoreset=True)
    
    banner = fr"""
{Fore.CYAN}
    _   _                      _       ___  ____  __  __ _______   _________ _    _ _____  ______ 
   | \ | |                    | |     / _ \|  _ \|  \/  |_   _\ \ / /__   __| |  | |  __ \|  ____|
   |  \| | ___ _   _ _ __ __ _| |    / /_\ | | | | \  / | | |  \ V /   | |  | |  | | |__) | |__   
   | . ` |/ _ \ | | | '__/ _` | |    |  _  | | | | |\/| | | |   > <    | |  | |  | |  _  /|  __|  
   | |\  |  __/ |_| | | | (_| | |    | | | | |_| | |  | |_| |_ / . \   | |  | |__| | | \ \| |____ 
   |_| \_|\___|\__,_|_|  \__,_|_|    \_| |_/____/|_|  |_|_____/_/ \_\  |_|   \____/|_|  \_\______|
{Style.RESET_ALL}
    """
    
    info = f"""
        {Fore.CYAN}                             Version: {version}{Style.RESET_ALL}
        {Fore.CYAN}            Authors: Joan Saurina Ricós, Albert Dominguez Mantes, 
                            Daniel Mas Montserrat, Alexander G. Ioannidis.{Style.RESET_ALL}
        {Fore.CYAN}            Help: https://github.com/AI-sandbox/neural-admixture{Style.RESET_ALL}
            """
    
    log.info("\n" + banner + info)

def main():
    print_neural_admixture_banner(__version__)
    arg_list = tuple(sys.argv)
    mode = arg_list[1]
    assert len(arg_list) > 1, 'Please provide either the argument "train" or "infer" to choose running mode.'
    
    if mode == "train":
        args = parse_train_args(arg_list[2:])
    elif mode == "infer":
        args = parse_infer_args(arg_list[2:])
    else:
        assert False, f'Unknown mode "{mode}". Please use "train" or "infer".'
    
    # VERIFY INPUT:
    assert args.threads > 0, "Please select a valid number of threads (>0)."
    assert args.seed >= 0, "Please select a valid seed (>=0)."
    assert args.num_gpus >= 0, "Number of GPUs must be >= 0."
    assert args.batch_size > 0, "Batch size must be > 0."
    if mode == "train":
        assert args.epochs > 0, "Number of epochs must be > 0."
        assert args.learning_rate > 0, "Learning rate must be > 0."
        assert args.hidden_size > 0, "Hidden size must be > 0."
        assert args.supervised_loss_weight >= 0, "Supervised loss weight must be >= 0."
        assert args.n_components > 0, "Number of components for SVD must be > 0."

        if args.k is not None:
            assert args.k > 1, "Please select K > 1."
            log.info(f"    Running on K = {args.k}.")
        elif args.min_k is not None and args.max_k is not None:
            assert args.min_k > 1, "min_k must be greater than 1."
            assert args.max_k > args.min_k, "max_k must be greater than min_k."
            log.info(f"    Running from K={args.min_k} to K={args.max_k}.")
        else:
            raise ValueError("Please provide either --k or both --min_k and --max_k.")

    if mode == "infer":
        assert args.batch_size > 0, "Batch size must be > 0."

    # CONTROL TIME:
    t0 = time.time()
    
    # CONTROL RESOURCES:
    os.environ["NUMEXPR_MAX_THREADS"] = str(args.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_MAX_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_MAX_THREADS"] = str(args.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_MAX_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MAX_JOBS"] = str(max(1, args.threads // 2))

    log.info(f"    Using {args.threads} threads...")
    
    #CONTROL OS:
    system = platform.system()
    if system == "Linux":
        log.info("    Operating system is Linux!")
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"
    elif system == "Darwin":
        log.info("    Operating system is Darwin (Mac OS)!")
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
    elif system == "Windows":
        log.info("    Operating system is Windows!")
        pass
    else:
        log.info(f"System not recognized: {system}")
        sys.exit(1)

    # CONTROL NUMBER OF DEVICES:
    max_devices = torch.cuda.device_count() if not torch.backends.mps.is_available() else 1
    if args.num_gpus > max_devices:
        log.warning(f"    Requested {args.num_gpus} GPUs, but only {max_devices} are available. Using {max_devices} GPUs.")
        num_gpus = max_devices
    else:
        num_gpus = args.num_gpus
    
    # CONTROL SEED:
    utils.set_seed(args.seed)
    
    # BEGIN TRAIN OF INFERENCE:
    if mode == 'train':
        from .src import main
        data, pops, N, M = utils.read_data(args.data_path, args.pops_path)
        log.info("")
        log.info("    Running SVD...")
        log.info("")
        V = RSVD(data, N, M, args.n_components, args.seed)
        data = torch.as_tensor(data, dtype=torch.uint8).share_memory_()
        
        if num_gpus>1:
            log.info("    Entering multi-GPU training...")
            mp.spawn(main.main, args=(args, num_gpus, data, V, pops, t0), nprocs=num_gpus)
        else:
            log.info("    Entering single-GPU or CPU training...")
            sys.exit(main.main(0, args, num_gpus, data, V, pops, t0))
    
    elif mode == 'infer':
        from .src import inference
        log.info("    Entering inference...")
        sys.exit(inference.main(args, t0))