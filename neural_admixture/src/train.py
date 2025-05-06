import argparse
import dask.array as da
import logging
import sys
import time
import torch
from pathlib import Path
from typing import List
from argparse import ArgumentError, ArgumentTypeError
from pathlib import Path

from . import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def fit_model(args: argparse.Namespace, trX: da.core.Array, device: torch.device, num_gpus: int,
            master: bool, has_missing: bool) -> None:
    """Wrapper function to start training
    """
    (epochs, batch_size, learning_rate, save_dir, activation_str, hidden_size, initialization, 
    n_components, name, seed, num_cpus) = (int(args.epochs), int(args.batch_size), float(args.learning_rate), args.save_dir, 
                                args.activation, int(args.hidden_size), args.initialization, 
                                int(args.n_components), args.name, int(args.seed), int(args.num_cpus))
        
    utils.set_seed(seed)
    
    K = int(args.k)
    P, Q = utils.train(initialization, device, K, seed, n_components, epochs, batch_size, learning_rate, trX, num_gpus, 
                            activation_str, hidden_size, master, num_cpus, has_missing)
    if master:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        utils.write_outputs(Q, name, K, save_dir, P)

    return

"""
def perform_cross_validation(args: argparse.Namespace, trX: da.core.Array, device: torch.device, num_gpus: int, 
                             master: bool) -> None:
    
    Perform cross-validation and log the results.

    Args:
        args: A namespace object containing command-line arguments.
        trX: Training data.
        device: A string representing the device ('cuda:0', 'cpu', etc.)
        num_gpus: Number of GPUs.
    
    if master:
        log.info(f'Performing {args.cv}-fold cross-validation...')
    cv_obj = KFold(n_splits=args.cv, random_state=args.seed, shuffle=True)
    cv_errs = []
    
    for tr_idx, val_idx in tqdm(cv_obj.split(trX), desc="Cross-Validation"):
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            trX_curr, valX_curr = trX[tr_idx], trX[val_idx]
        
        for k in range (args.k_range[0], args.k_range[1]):
            loglikelihood = fit_model(args, trX_curr, valX_curr, device, num_gpus, master, k=k)
            cv_errs.append(loglikelihood)
    
    cv_errs_reduced = utils.process_cv_loglikelihood(cv_errs)
    
    if master:
        for _, row in cv_errs_reduced.iterrows():
            log.info(f"CV error (K={int(row['K'])}): {row['cv_error_mean']:.5f} ± {row['cv_error_std']:.3f}")
    
    utils.save_cv_error_plot(cv_errs_reduced, args.save_dir)
"""

def main(rank: int, argv: List[str], num_gpus):
    """Training entry point
    """
    utils.ddp_setup('begin', rank, num_gpus)
    master = rank == 0
    
    try:
        if any(arg in argv for arg in ['-h', '--help']):
            if master:
                args = utils.parse_train_args(argv)
            return
        args = utils.parse_train_args(argv)
        
        if num_gpus>0:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{int(rank)}')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        
        if master:
            log.info(f"    There are {args.num_cpus} CPUs and {num_gpus} GPUs available for this execution.")
            log.info("")
            log.info(f"    Running on K = {args.k}.")
            log.info("")
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
        t0 = time.time()
        trX, has_missing = utils.read_data(args.data_path, master)

        #if args.cv is not None:
        #    perform_cross_validation(args, trX, device, num_gpus, master)   
        
        fit_model(args, trX, device, num_gpus, master, has_missing)
        
        if master:
            t1 = time.time()
            log.info("")
            log.info(f"    Total elapsed time: {t1-t0:.2f} seconds.")
            log.info("")
        
        logging.shutdown()
        utils.ddp_setup('end', rank, num_gpus)
    
    except (ArgumentError, ArgumentTypeError) as e:
        if master:
            log.error(f"    Error parsing arguments")
        logging.shutdown()
        utils.ddp_setup('end', rank, num_gpus)
        if master:
            raise e
        
    except Exception as e:
        if master:
            log.error(f"    Unexpected error")
        logging.shutdown()
        utils.ddp_setup('end', rank, num_gpus)        
        if master:
            raise e
        
if __name__ == '__main__':
    main(sys.argv[1:])