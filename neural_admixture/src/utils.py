import argparse
import configargparse
import logging
import numpy as np
import os
import sys
import torch
import wandb

import dask.array as da
from pathlib import Path
from typing import List, Tuple, Union

from .snp_reader import SNPReader
from ..model.neural_admixture import NeuralAdmixture

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def parse_train_args(argv: List[str]):
    """Training arguments parser
    """
    parser = configargparse.ArgumentParser(prog='neural-admixture train',
                                           description='Rapid population clustering with autoencoders - training mode',
                                           config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--learning_rate', required=False, default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--max_epochs', required=False, type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--initialization', required=False, type=str, default = 'pcarchetypal',
                        choices=['pretrained', 'pckmeans', 'supervised', 'pcarchetypal'],
                        help='Decoder initialization (overriden if supervised)')
    parser.add_argument('--optimizer', required=False, default='adam', type=str, choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--save_every', required=False, default=10, type=int, help='Save checkpoint every this number of epochs')
    parser.add_argument('--l2_penalty', required=False, default=0.0005, type=float, help='L2 penalty on encoder weights')
    parser.add_argument('--activation', required=False, default='gelu', type=str, choices=['relu', 'tanh', 'gelu'], help='Activation function for encoder layers')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    parser.add_argument('--k', required=False, type=int, help='Number of populations/clusters (single-head)')
    parser.add_argument('--min_k', required=False, type=int, help='Minimum number of populations/clusters (multi-head)')
    parser.add_argument('--max_k', required=False, type=int, help='Maximum number of populations/clusters (multi-head)')
    parser.add_argument('--hidden_size', required=False, default=64, type=int, help='Dimension of first projection in encoder')
    parser.add_argument('--init_file', required=False, type=str, help='File name of precomputed initialization weights to load from/save to')
    parser.add_argument('--freeze_decoder', action='store_true', default=False, help='If specified, will freeze decoder weights during training (only use if providing a computed initialization!)')
    parser.add_argument('--supervised', action='store_true', default=False, help='If specified, will run the supervised version')
    parser.add_argument('--validation_data_path', required=False, default='', type=str, help='Path containing the validation data')
    parser.add_argument('--populations_path', required=False, default='', type=str, help='Path containing the main data populations')
    parser.add_argument('--validation_populations_path', required=False, default='', type=str, help='Path containing the validation data populations')
    parser.add_argument('--wandb_log', required=False, action='store_true', default=False, help='Whether to log to wandb or not')
    parser.add_argument('--wandb_user', required=False, type=str, help='wandb user')
    parser.add_argument('--wandb_project', required=False, type=str, help='wandb project')
    parser.add_argument('--pca_path', required=False, type=str, help='Path containing PCA object, used for plots and to store checkpoints')
    parser.add_argument('--pca_components', required=False, type=int, default=2, help='Number of components to use for the PCKMeans initialization')
    parser.add_argument('--tol', required=False, type=float, default=1e-6, help='Convergence criterion: will stop when difference in objective function between two iterations is smaller than this')
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name')
    parser.add_argument('--batch_size', required=False, default=400, type=int, help='Batch size')
    parser.add_argument('--supervised_loss_weight', required=False, default=0.05, type=float, help='Weight given to the supervised loss')
    return parser.parse_args(argv)

def parse_infer_args(argv: List[str]):
    """Inference arguments parser
    """
    parser = argparse.ArgumentParser(prog='neural-admixture infer',
                                     description='Rapid population clustering with autoencoders - inference mode')
    parser.add_argument('--out_name', required=True, type=str, help='Name used to output files on inference mode')
    parser.add_argument('--save_dir', required=True, type=str, help='Load model from this directory')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Trained experiment/model name')
    parser.add_argument('--batch_size', required=False, default=400, type=int, help='Batch size')
    return parser.parse_args(argv)

def initialize_wandb(run_name: str, trX: da.core.Array, valX: da.core.Array, args: argparse.Namespace, out_path: str, silent: bool=True) -> str:
    """Initializes wandb project run

    Args:
        run_name (str): run name.
        trX (da.core.Array): Dask array containing training data.
        valX (da.core.Array): Dask array containing validation data.
        args (argparse.Namespace): parsed arguments.
        out_path (str): results output path.
        silent (bool, optional): if set, skips wandb log messages. Defaults to True.

    Returns:
        str: run name.
    """
    if run_name is None:
        log.warn('Run name for wandb not specified. Skipping.')
        return None
    if silent:
        os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=args.wandb_project,
               entity=args.wandb_user,
               name=run_name,
               config=args,
               settings=wandb.Settings(start_method='fork')
            )
    wandb.config.update({'train_samples': trX.shape[0],
                         'val_samples': valX.shape[0] if valX is not None else '',
                         'SNPs': trX.shape[1],
                         'out_path': out_path})
    return run_name

def read_data(tr_file: str, val_file: str=None, tr_pops_f: str=None, val_pops_f: str=None) -> Tuple[da.core.Array, Union[None, da.core.Array], Union[None, List[str]], Union[None, List[str]]]:
    """Read data in any compatible format

    Args:
        tr_file (str): denotes the path of the main data file
        val_file (str, optional): denotes the path of the validation data file. Defaults to None.
        tr_pops_f (str, optional): denotes the path containing the main populations file. Defaults to None.
        val_pops_f (str, optional): denotes the path containing the validation populations file. Defaults to None.

    Returns:
        _type_: _description_
    """
    ''' 
    tr_file: string denoting the path of the main data file. The format of this file must be either HDF5 or VCF.
    val_file: optional. String denoting the path of the validation data file. The format of this file must be either HDF5 or VCF.
    tr_pops_f: optional. String denoting the path containing the main populations file. It must be a plain txt file where each row is a number specifying the population of the corresponding sample, as in ADMIXTURE.
    val_pops_f: optional. String denoting the path containing the validation populations file. It must be a plain txt file where each row is a number specifying the population of the corresponding sample, as in ADMIXTURE.
    '''
    tr_pops, val_pops = None, None
    log.info('Reading data...')
    snp_reader = SNPReader()
    tr_snps = snp_reader.read_data(tr_file)
    val_snps = snp_reader.read_data(val_file) if val_file else None
    if tr_pops_f:
        with open(tr_pops_f, 'r') as fb:
            tr_pops = [p.strip() for p in fb.readlines()]
    if val_pops_f:
        with open(val_pops_f, 'r') as fb:
            val_pops = [p.strip() for p in fb.readlines()]
    validate_data(tr_snps, tr_pops, val_snps, val_pops)
    log.info(f'Data contains {tr_snps.shape[0]} samples and {tr_snps.shape[1]} SNPs.')
    if val_snps is not None:
        log.info(f'Validation data contains {val_snps.shape[0]} samples and {val_snps.shape[1]} SNPs.')
    log.info('Data loaded.')
    return tr_snps, tr_pops, val_snps, val_pops

def validate_data(tr_snps: da.core.Array, tr_pops: Union[None, List[str]], val_snps: Union[None, da.core.Array], val_pops: Union[None, List[str]]):
    """Data sanity check

    Args:
        tr_snps (da.core.Array): Dask array containing training data.
        tr_pops (Union[None, List[str]]): list containing training population labels.
        val_snps (Union[None, da.core.Array]): Dask array containing validation data.
        val_pops (Union[None, List[str]]): list containing validation population labels.
    """
    assert not (val_snps is None and val_pops is not None), 'Populations were specified for validation data, but no SNPs were specified.'
    if tr_pops is not None:
        assert tr_snps.shape[0] == len(tr_pops), f'Number of samples in data and population file does not match: {len(tr_snps)} vs {len(tr_pops)}.'
    if val_snps is not None:
        assert tr_snps.shape[1] == val_snps.shape[1], f'Number of SNPs in training and validation data does not match: {tr_snps.shape[1]} vs {val_snps.shape[1]}.'
        if val_pops is not None:
            assert val_snps.shape[0] == len(val_pops), f'Number of samples in validation data and validation population file does not match: {len(tr_snps)} vs {len(tr_pops)}.'
    return

def get_model_predictions(model: NeuralAdmixture, data: da.core.Array, bsize: int, device: torch.device) -> List[np.ndarray]:
    """Helper function to run inference on data

    Args:
        model (NeuralAdmixture): trained model object.
        data (da.core.Array): Dask array containing data to get results from.
        bsize (int): batch size.
        device (torch.device): torch device.

    Returns:
        _type_: _description_
    """
    model.to(torch.device(device))
    outs = [torch.tensor([]) for _ in range(len(model.ks))]
    model.eval()
    with torch.inference_mode():
        for X, _ in model.batch_generator(data, bsize, shuffle=False):
            X = X.to(device)
            out = model(X, True)
            for j in range(len(model.ks)):
                outs[j] = torch.cat((outs[j], out[j].detach().cpu()), axis=0)
    return [out.cpu().numpy() for out in outs]

def write_outputs(model: NeuralAdmixture, trX: da.core.Array, valX: Union[da.core.Array, None],
                  bsize: int, device: torch.device, run_name: str, out_path: str, only_Q: bool=False) -> int:
    """Helper function to write Q and P matrices to disk

    Args:
        model (NeuralAdmixture): trained model object.
        trX (da.core.Array): Dask array containing training data.
        valX (Union[da.core.Array, None]): Dask array containing validation data.
        bsize (int): batch size.
        device (torch.device): torch device.
        run_name (str): run name.
        out_path (str): output directory path.
        only_Q (bool, optional): if set, only the Q matrix will be written. Defaults to False.

    Returns:
        int: status
    """
    out_path = Path(out_path)
    if not only_Q:
        for dec in model.decoders.decoders:
            w = 1-dec.weight.data.cpu().numpy()
            k = dec.in_features
            np.savetxt(out_path/f"{run_name}.{k}.P", w, delimiter=' ')
    tr_preds = get_model_predictions(model, trX, bsize, device)
    for i, k in enumerate(model.ks):
        np.savetxt(out_path/f"{run_name}.{k}.Q", tr_preds[i], delimiter=' ')
    if valX is not None:
        val_preds = get_model_predictions(model, valX, bsize, device)
        for i, k in enumerate(model.ks):
            np.savetxt(out_path/f"{run_name}_validation.{k}.Q", val_preds[i], delimiter=' ')
    return 0

