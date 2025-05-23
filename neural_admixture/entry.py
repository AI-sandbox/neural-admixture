import logging
import sys
import os
import torch
import torch.multiprocessing as mp
import platform
import sys

from ._version import __version__

from .src import utils
from .src.svd import RSVD

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

from colorama import init, Fore, Style

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
    
    assert len(arg_list) > 1, 'Please provide either the argument "train" or "infer" to choose running mode.'
    
    # CONTROL NUMBER OF THREADS:
    if '--num_cpus' in arg_list:
        num_cpus_index = arg_list.index('--num_cpus') + 1
        if num_cpus_index < len(arg_list):
            num_cpus = int(arg_list[num_cpus_index])
            num_threads = num_cpus//2
    else:
        num_cpus = 1
        num_threads = 1
    
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MKL_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_MAX_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_MAX_THREADS"] = "1"
    
    log.info(f"    There are {num_cpus} CPU's available for this execution. Hence, using {num_threads} threads.")
    
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
    num_gpus = 0
    if '--num_gpus' in arg_list:
        num_gpus_index = arg_list.index('--num_gpus') + 1
        if num_gpus_index < len(arg_list):
            num_gpus = int(arg_list[num_gpus_index])
    
    max_devices = torch.cuda.device_count()
    if num_gpus > max_devices:
        log.warning(f"    Requested {num_gpus} GPUs, but only {max_devices} are available. Using {max_devices} GPUs.")
        num_gpus = max_devices
    
    # CONTROL SEED:
    seed = int(arg_list[arg_list.index('--seed') + 1]) if '--seed' in arg_list else 42
    utils.set_seed(seed)
    
    # BEGIN TRAIN OF INFERENCE:
    if sys.argv[1]=='train':
        from .src import main
        
        data_path = arg_list[arg_list.index('--data_path') + 1]
        pops_path = arg_list[arg_list.index('--pops_path') + 1] if '--pops_path' in arg_list else None
        n_components = int(arg_list[arg_list.index('--n_components') + 1]) if '--n_components' in arg_list else 8
        data, pops, N, M = utils.read_data(data_path, pops_path)
        log.info("")
        log.info("    Running SVD...")
        log.info("")
        V = RSVD(data, N, M, n_components, seed)
        data = torch.as_tensor(data, dtype=torch.uint8).share_memory_()
        
        if num_gpus>1:
            log.info("    Entering multi-GPU training...")
            mp.spawn(main.main, args=(arg_list[2:], num_gpus, data, V, pops), nprocs=num_gpus)
        else:
            log.info("    Entering single-GPU or CPU training...")
            sys.exit(main.main(0, arg_list[2:], num_gpus, data, V, pops))
    
    elif sys.argv[1]=='infer':
        from .src import inference
        log.info("    Entering inference...")
        sys.exit(inference.main(arg_list[2:]))
    
    else:
        log.error(f'    Invalid argument {arg_list[1]}. Please run either "neural-admixture train" or "neural-admixture infer"')
        sys.exit(1)