import logging
import sys
import os
import torch
import torch.multiprocessing as mp

from ._version import __version__

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def print_neural_admixture_banner(version: str="2.0") -> None:
    """
    Display the Neural Admixture banner with version and author information.

    Args:
        version (str): Version number to display. Defaults to "2.0".

    Returns:
        None
    """
    banner = r"""
    _   _                      _       ___  ____  __  __ _______   _________ _    _ _____  ______ 
   | \ | |                    | |     / _ \|  _ \|  \/  |_   _\ \ / /__   __| |  | |  __ \|  ____|
   |  \| | ___ _   _ _ __ __ _| |    / /_\ | | | | \  / | | |  \ V /   | |  | |  | | |__) | |__   
   | . ` |/ _ \ | | | '__/ _` | |    |  _  | | | | |\/| | | |   > <    | |  | |  | |  _  /|  __|  
   | |\  |  __/ |_| | | | (_| | |    | | | | |_| | |  | |_| |_ / . \   | |  | |__| | | \ \| |____ 
   |_| \_|\___|\__,_|_|  \__,_|_|    \_| |_/____/|_|  |_|_____/_/ \_\  |_|   \____/|_|  \_\______|
                                                                                          
    """
    
    info = f"""
    Version: {version}
    Authors: Joan Saurina Ricós, Albert Dominguez Mantes, 
            Daniel Mas Montserrat and Alexander G. Ioannidis.
    
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
    
    #str(num_threads)
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MKL_MAX_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OMP_MAX_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_MAX_THREADS"] = "1"
    
    log.info(f"    There are {num_cpus} CPU's available for this execution. Hence, using {num_threads} threads.")

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
    
    # BEGIN TRAIN OF INFERENCE:
    if sys.argv[1]=='train':
        from .src import train

        if num_gpus>1:
            log.info("    Entering multi-GPU training...")
            mp.spawn(train.main, args=(arg_list[2:], num_gpus), nprocs=num_gpus)
        else:
            log.info("    Entering single-GPU or CPU training...")
            sys.exit(train.main(0, arg_list[2:], num_gpus))
    
    elif sys.argv[1]=='infer':
        from .src import inference
        log.info("    Entering inference...")
        sys.exit(inference.main(arg_list[2:]))
    
    else:
        log.error(f'    Invalid argument {arg_list[1]}. Please run either "neural-admixture train" or "neural-admixture infer"')
        sys.exit(1)