import allel
import argparse
import h5py
import gc
import logging
import numpy as np
import os
import sys
import torch
import wandb
from pandas_plink import read_plink

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
def parse_args(train=True):
    parser = argparse.ArgumentParser()
    if train:
        parser.add_argument('--learning_rate', required=False, default=0.0001, type=float, help='Learning rate')
        parser.add_argument('--max_epochs', required=False, type=int, default=50, help='Maximum number of epochs')
        parser.add_argument('--decoder_init', required=False, type=str, default = 'pckmeans', choices=['random', 'mean_SNPs', 'mean_random', 'kmeans',
                                                                                                       'minibatch_kmeans', 'kmeans++', 'binomial',
                                                                                                       'pca', 'admixture', 'pckmeans', 'supervised',
                                                                                                       'tsvdkmeans'],
                                                                                                       help='Decoder initialization (overriden if supervised)')
        parser.add_argument('--optimizer', required=False, default='adam', type=str, choices=['adam', 'sgd'], help='Optimizer')
        parser.add_argument('--save_every', required=False, default=50, type=int, help='Save every this number of epochs')
        parser.add_argument('--l2_penalty', required=False, default=0.01, type=float, help='L2 penalty on encoder weights')
        parser.add_argument('--display_logs', required=False, type=int, default=1, choices=[0, 1], help='Whether to display logs during training or not')
        parser.add_argument('--activation', required=False, default='relu', type=str, choices=['relu', 'tanh'], help='Activation function for deep encoder layers')
        parser.add_argument('--wandb_log', required=False, default=0, type=int, choices=[0, 1], help='Whether to log to wandb or not')
        parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
        parser.add_argument('--k', required=False, type=int, help='Minimum number of clusters for multihead admixture')
        parser.add_argument('--min_k', required=False, type=int, help='Minimum number of clusters for multihead admixture')
        parser.add_argument('--max_k', required=False, type=int, help='Maximum number of clusters for multihead admixture')
        parser.add_argument('--shuffle', required=False, default=1, type=int, choices=[0, 1], help='Whether to shuffle the training data at every epoch')
        parser.add_argument('--hidden_size', required=False, default=512, type=int, help='Hidden size in encoder and non-linear decoder')
        parser.add_argument('--linear', required=False, default=1, type=int, choices=[0, 1], help='Whether to use a linear decoder or not')
        parser.add_argument('--init_file', required=False, type=str, help='File name of precomputed initialization weights to load from/save to')
        parser.add_argument('--freeze_decoder', action='store_true', default=False, help='Whether to freeze linear decoder weights')
        parser.add_argument('--supervised', action='store_true', default=False, help='Whether to use the supervised version or not')
        parser.add_argument('--dataset', required=False, type=str, help='Dataset to be used (to replicate experiments)')
        parser.add_argument('--validation_data_path', required=False, default='', type=str, help='Path containing the validation data')
        parser.add_argument('--populations_path', required=False, default='', type=str, help='Path containing the main data populations')
        parser.add_argument('--validation_populations_path', required=False, default='', type=str, help='Path containing the validation data populations')
        parser.add_argument('--wandb_user', required=False, type=str, help='wandb user')
        parser.add_argument('--wandb_project', required=False, type=str, help='wandb project')
        parser.add_argument('--pca_path', required=False, type=str, help='Path containing PCA object, used for plots')
        parser.add_argument('--pca_components', required=False, type=int, default=2, help='Number of components to use for the PCKMeans initialization')
        parser.add_argument('--tol', required=False, type=float, default=1e-6, help='Convergence criterion: will stop when difference in objective function between two iterations is smaller than this.')

    else:
        parser.add_argument('--out_name', required=True, type=str, help='Name used to output files on inference mode.')
    parser.add_argument('--save_dir', required=True, type=str, help='{} this directory'.format('Save model in' if train else 'Load model from'))
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name')
    parser.add_argument('--batch_size', required=False, default=200, type=int, help='Batch size')
    parser.add_argument('-i', action='store_true', required=False, help='Dummy flag used in the entry script.')
    return parser.parse_args()

def initialize_wandb(run_name, trX, valX, args, out_path, silent=True):
    if run_name is None:
        log.warn('Run name for wandb not specified. Skipping logging.')
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
                         'out_path': out_path,
                         'averaged_parents': True,
                         'sum_parents': False})
    return run_name

def read_data(tr_file, val_file=None, tr_pops_f=None, val_pops_f=None):
    '''
    tr_file: string denoting the path of the main data file. The format of this file must be either HDF5 or VCF.
    val_file: optional. String denoting the path of the validation data file. The format of this file must be either HDF5 or VCF.
    tr_pops_f: optional. String denoting the path containing the main populations file. It must be a plain txt file where each row is a number specifying the population of the corresponding sample, as in ADMIXTURE.
    val_pops_f: optional. String denoting the path containing the validation populations file. It must be a plain txt file where each row is a number specifying the population of the corresponding sample, as in ADMIXTURE.
    '''
    tr_snps, tr_pops, val_snps, val_pops = None, None, None, None
    # Training data
    if tr_file.endswith('.h5') or tr_file.endswith('.hdf5'):
        log.info('Input format is HDF5.')
        f_tr = h5py.File(tr_file, 'r')
        tr_snps = f_tr['snps']
    elif tr_file.endswith('.vcf') or tr_file.endswith('.vcf.gz'):
        log.info('Input format is VCF.')
        log.info('Reading data...')
        f_tr = allel.read_vcf(tr_file)
        log.info('Processing data...')
        tr_snps = np.sum(f_tr['calldata/GT'], axis=2).T/2
    elif tr_file.endswith('.bed'):
        log.info('Input format is BED.')
        log.info('Reading data...')
        _, _, G = read_plink('.'.join(tr_file.split('.')[:-1]))
        log.info('Processing data...')
        tr_snps = ((2-G).T/2).compute()
        del G
        gc.collect()
    else:
        log.error('Unrecognized file format. Make sure file ends with .h5 | .hdf5 | .vcf | .vcf.gz | .bed')
        sys.exit(1)

    # Validation data
    if val_file:
        if val_file.endswith('.h5') or val_file.endswith('.hdf5'):
            log.info('Validation input format is HDF5.')
            f_val = h5py.File(val_file, 'r')
            val_snps = f_val['snps']
        elif val_file.endswith('.vcf') or val_file.endswith('.vcf.gz'):
            log.info('Validation input format is VCF.')
            log.info('Reading validation data...')
            f_val = allel.read_vcf(val_file)
            log.info('Processing validation data...')
            val_snps = np.sum(f_val['calldata/GT'], axis=2).T/2
        elif tr_file.endswith('.bed'):
            log.info('Validation input format is BED.')
            log.info('Reading validation data...')
            _, _, G = read_plink('.'.join(val_file.split('.')[:-1]))
            log.info('Processing validation data...')
            val_snps = ((2-G).T/2).compute()
            del G
            gc.collect()
        else:
            log.error('Unrecognized validation file format. Make sure file ends with .h5 | .hdf5 | .vcf | .vcf.gz | .bed')
            sys.exit(1)
    if tr_pops_f:
        with open(tr_pops_f, 'r') as fb:
            tr_pops = fb.readlines()
    if val_pops_f:
        with open(val_pops_f, 'r') as fb:
            val_pops = fb.readlines()
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
