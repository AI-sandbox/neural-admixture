import argparse
import dask
import dask.array as da
import logging
import pandas as pd
import sys
import torch
import torch.nn as nn
import wandb
from codetiming import Timer
from pathlib import Path
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Union

from ..model.neural_admixture import NeuralAdmixture
from ..model.switchers import Switchers
from . import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def fit_model(trX: da.core.Array, args: argparse.Namespace, valX: Union[None, da.core.Array]=None,
              trY: Union[None, List[str]]=None, valY: Union[None, List[str]]=None,
              dry_run: bool=False) -> Tuple[NeuralAdmixture, torch.device, List[Dict[int, float]]]:
    """Wrapper function to start training

    Args:
        trX (da.core.Array): Dask array containing training data.
        args (argparse.Namespace): parsed argument from CLI.
        valX (Union[None, da.core.Array], optional): Dask array containing validation data. Defaults to None.
        trY (Union[None, List[str]], optional): list containing training labels. Defaults to None.
        valY (Union[None, List[str]], optional): list containing validation labels. Defaults to Union[None, List[str]]=None.
        dry_run (bool, optional): whether to run a dry run (no output is written to disk). Defaults to False.

    Returns:
        Tuple[NeuralAdmixture, torch.device, List[Dict[int, float]]]: instantiated model object along with device and validation error.
    """
    switchers = Switchers.get_switchers()
    num_max_epochs = args.max_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    save_dir = args.save_dir
    optimizer_str = args.optimizer
    save_every = args.save_every
    l2_penalty = args.l2_penalty
    activation_str = args.activation
    hidden_size = args.hidden_size
    freeze_decoder = bool(args.freeze_decoder)
    init_file = args.init_file
    supervised = bool(args.supervised)
    supervised_loss_weight = float(args.supervised_loss_weight)
    decoder_init = args.initialization if not supervised else 'supervised'
    n_components = int(args.pca_components)
    tol = float(args.tol)
    warmup_epochs = int(args.warmup_epochs)
    name = args.name
    if args.k is not None:
        Ks = [int(args.k)]
    elif args.min_k is not None and args.max_k is not None:
        Ks = [i for i in range(args.min_k, args.max_k+1)]
    else:
        log.error('Either --k (single-head) or --min_k and --max_k (multi-head) must be provided.')
        sys.exit(1)
    assert not (supervised and len(Ks) != 1), 'Supervised version is currently only available on a single head'
    seed = args.seed
    log_to_wandb = bool(args.wandb_log)
    run_name = name
    if not dry_run:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    log.info(f'Job args: {args}')
    log.info('Will use GPU.' if torch.cuda.is_available() else 'Will use Apple Metal acceleration.' if torch.backends.mps.is_available() else 'No GPUs available.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if log_to_wandb and not dry_run:
        utils.initialize_wandb(run_name, trX, valX, args, save_dir)
    save_path = f'{save_dir}/{run_name}.pt'
    torch.manual_seed(seed)
    # Initialization
    log.info('Initializing...')
    if init_file is None and decoder_init != "pretrained":
        if not dry_run:
            log.warning(f'Initialization filename not provided. Going to store it to {Path(save_dir)/run_name}.pkl')
        init_file = f'{run_name}.pkl'
    init_path = f'{Path(save_dir)/init_file}' if decoder_init != "pretrained" else init_file if not dry_run else None
    P_init, Q_inits = switchers['initializations'][decoder_init](
        trX,
        trY,
        Ks,
        seed,
        init_path,
        run_name,
        n_components,
        batch_size)
    activation = switchers['activations'][activation_str](0)
    log.info(f'Variants: {trX.shape[1]}')
    model = NeuralAdmixture(Ks, trX.shape[1], P_init=P_init,
                                lambda_l2=l2_penalty,
                                encoder_activation=activation,
                                hidden_size=hidden_size,
                                freeze_decoder=freeze_decoder,
                                supervised=supervised,
                                supervised_loss_weight=supervised_loss_weight)
    model.to(device)
    if log_to_wandb:
        wandb.watch(model, log='all', log_freq=1000)

    # Optimizer
    optimizer = switchers['optimizers'][optimizer_str](filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    log.info('Optimizer successfully loaded.')

    loss_f = nn.BCELoss(reduction='mean')
    log.info('Going to train {} head{}: {}.'.format(len(Ks), 's' if len(Ks) > 1 else '', f'K={Ks[0]}' if len(Ks) == 1 else f'K={Ks[0]} to K={Ks[-1]}'))
    # Fitting
    log.info('Fitting...')
    t = Timer()
    t.start()
    actual_num_epochs = model.launch_training(trX, optimizer, loss_f, num_max_epochs, device, valX=valX,
                       batch_size=batch_size, save_every=save_every,
                       save_path=save_path, trY=trY, valY=valY,
                       seed=seed, log_to_wandb=log_to_wandb, tol=tol,
                       warmup_epochs=warmup_epochs, Q_inits=Q_inits)
    deviances = utils.compute_deviances(model, valX, batch_size, device) if valX is not None else -1
    elapsed_time = t.stop()
    if not dry_run:
        if log_to_wandb:
            wandb.run.summary['total_elapsed_time'] = elapsed_time
            wandb.run.summary['avg_epoch_time'] = elapsed_time/actual_num_epochs
        torch.save(model.state_dict(), save_path)
        model.save_config(run_name, save_dir)
    log.info('Optimization process finished.')
    return model, device, deviances

def main(argv: List[str]):
    """Training entry point
    """
    args = utils.parse_train_args(argv)
    tr_file, val_file = args.data_path, args.validation_data_path
    # assert not (val_file and args.cv is not None), 'Cross-validation not available when validation data path is provided.'
    tr_pops_f, val_pops_f = args.populations_path, args.validation_populations_path
    trX, trY, valX, valY = utils.read_data(tr_file, val_file, tr_pops_f, val_pops_f, imputation=args.imputation)
    """
    if args.cv is not None:
        log.info(f'Performing {args.cv}-fold cross-validation...')
        cv_obj = KFold(n_splits=args.cv, random_state=args.seed, shuffle=True)
        cv_errs = []
        # TODO: TQDM or improve logging for cross-validation
        for tr_idx, val_idx in cv_obj.split(trX):
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                trX_curr, valX_curr = trX[tr_idx], trX[val_idx]
            trY_curr = [trY[idx] for idx in tr_idx] if trY is not None else None
            valY_curr = [trY[idx] for idx in val_idx] if trY is not None else None
            model, device, deviance_curr = fit_model(trX_curr, args, valX_curr, trY_curr, valY_curr, dry_run=True)
            cv_errs += [deviance_curr]

        # TODO: wrap this in a function
        cv_errs = pd.DataFrame.from_records(cv_errs)
        cv_errs_reduced = pd.DataFrame(cv_errs.mean())
        cv_errs_reduced["K"] = cv_errs_reduced.index.copy()
        cv_errs_reduced.rename(columns={0: "cv_error_mean"}, inplace=True)
        cv_errs_reduced["cv_error_std"] = cv_errs.std()
        cv_errs_reduced = cv_errs_reduced.sort_values("K")
    """
    model, device, _ = fit_model(trX, args, valX, trY, valY)
    log.info('Computing divergences...')
    model.display_divergences()
#     if args.cv:
#         for _, row in cv_errs_reduced.iterrows():
#             log.info(f"CV error (K={int(row['K'])}): {row['cv_error_mean']:.5f} ± {row['cv_error_std']:.3f}")
    log.info('Writing outputs...')
    utils.write_outputs(model, trX, valX, args.batch_size, device, args.name, args.save_dir)
    log.info('Exiting...')
    logging.shutdown()
    return 0
