import argparse
import logging
import sys
import time
import torch
import numpy as np

from pathlib import Path
from typing import List
from argparse import ArgumentError, ArgumentTypeError
from pathlib import Path

from . import utils
from ..model.train import train

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def fit_model(args: argparse.Namespace, data: torch.Tensor, device: torch.device, num_gpus: int,
            master: bool, V: np.ndarray, pops: np.ndarray) -> None:
    """
    Wrapper function to start training
    """
    (epochs, batch_size, learning_rate, save_dir, hidden_size, name, seed, n_components) = (int(args.epochs), int(args.batch_size), float(args.learning_rate), args.save_dir, 
                                                                                        int(args.hidden_size), args.name, int(args.seed), int(args.n_components))
            
    if args.k is not None:
        K = int(args.k)
        min_k = None
        max_k = None
    else:
        min_k = int(args.min_k)
        max_k = int(args.max_k)
        K = None

    Ps, Qs, model = train(epochs, batch_size, learning_rate, K, seed, data, device, num_gpus, hidden_size, master, V, pops, min_k, max_k, n_components)
    
    if master:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = f'{save_dir}/{name}.pt'
        state_dict = {key: value for key, value in model.state_dict().items() if not key.startswith('decoders')}
        torch.save(state_dict, save_path)
        model.save_config(name, save_dir)
        utils.write_outputs(Qs, name, K, min_k, max_k, save_dir, Ps)

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

def main(rank: int, args: List[str], num_gpus: int, data: torch.Tensor, V: np.ndarray, pops: np.ndarray, t0: float):
    """
    Training entry point
    """
    utils.ddp_setup('begin', rank, num_gpus)
    master = rank == 0
    
    try:
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
            log.info(f"    There are {args.threads} threads and {num_gpus} GPUs available for this execution.")
            log.info("")
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            
        #if args.cv is not None:
        #    perform_cross_validation(args, trX, device, num_gpus, master)   
        
        fit_model(args, data, device, num_gpus, master, V, pops)
        
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