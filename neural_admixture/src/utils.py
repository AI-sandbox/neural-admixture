import argparse
import gc
import logging
import numpy as np
import os
import sys
import time
import torch
import wandb
from itertools import permutations

from src.snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'infer'], help='Choose between modes.')
    parser.add_argument('--learning_rate', required=False, default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--max_epochs', required=False, type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--initialization', required=False, type=str, default = 'pckmeans',
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
    return parser.parse_args()

def parse_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'infer'], help='Choose between modes')
    parser.add_argument('--out_name', required=True, type=str, help='Name used to output files on inference mode')
    parser.add_argument('--save_dir', required=True, type=str, help='Load model from this directory')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Trained experiment/model name')
    parser.add_argument('--batch_size', required=False, default=400, type=int, help='Batch size')
    parser.add_argumnet('--out_dir', required=False, type=str, help='Path to save the output. If not given sets to --save_dir.')
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.save_dir
    return args

def initialize_wandb(run_name, trX, valX, args, out_path, silent=True):
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

def read_data(tr_file, val_file=None, tr_pops_f=None, val_pops_f=None):
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

def validate_data(tr_snps, tr_pops, val_snps, val_pops):
    assert not (val_snps is None and val_pops is not None), 'Populations were specified for validation data, but no SNPs were specified.'
    if tr_pops is not None:
        assert len(tr_snps) == len(tr_pops), f'Number of samples in data and population file does not match: {len(tr_snps)} vs {len(tr_pops)}.'
    if val_snps is not None:
        assert tr_snps.shape[1] == val_snps.shape[1], f'Number of SNPs in training and validation data does not match: {tr_snps.shape[1]} vs {val_snps.shape[1]}.'
        if val_pops is not None:
            assert len(val_snps) == len(val_pops), f'Number of samples in validation data and validation population file does not match: {len(tr_snps)} vs {len(tr_pops)}.'
    return

def get_model_predictions(model, data, bsize, device):
    model.to(torch.device(device))
    outs = [torch.tensor([]) for _ in range(len(model.ks))]
    model.eval()
    with torch.no_grad():
        for i, (X, _) in enumerate(model._batch_generator(data, bsize, shuffle=False)):
            X = X.to(device)
            out = model(X, True)
            for j in range(len(model.ks)):
                outs[j] = torch.cat((outs[j], out[j].detach().cpu()), axis=0)
    return [out.cpu().numpy() for out in outs]

def write_outputs(model, trX, valX, bsize, device, run_name, out_path, only_Q=False):
    if out_path.endswith('/'):
        out_path = out_path[:-1]
    if not only_Q:
        for dec in model.decoders.decoders:
            w = 1-dec.weight.data.cpu().numpy()
            k = dec.in_features
            np.savetxt(f'{out_path}/{run_name}.{k}.P', w, delimiter=' ')
    tr_preds = get_model_predictions(model, trX, bsize, device)
    for i, k in enumerate(model.ks):
        np.savetxt(f'{out_path}/{run_name}.{k}.Q', tr_preds[i], delimiter=' ')
    if valX is not None:
        val_preds = get_model_predictions(model, valX, bsize, device)
        for i, k in enumerate(model.ks):
            np.savetxt(f'{out_path}/{run_name}_validation.{k}.Q', val_preds[i], delimiter=' ')
    return 0

