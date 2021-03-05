import argparse
import h5py
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import plots
import torch
import wandb
from admixture_ae import AdmixtureAE
from codetiming import Timer
from switchers import Switchers
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

os.environ['WANDB_SILENT'] = 'true'


def read_data(window_size=0):
    log.info('Reading data...')
    f_tr = h5py.File(f'/home/usuaris/imatge/albert.dominguez/neural-admixture/data/chr22/prepared/train{window_size}.h5', 'r')
    f_val = h5py.File(f'/home/usuaris/imatge/albert.dominguez/neural-admixture/data/chr22/prepared/valid{window_size}.h5', 'r')
    return f_tr['snps'], f_tr['populations'], f_val['snps'], f_val['populations']

def fit_model(trX, valX, args):
    switchers = Switchers.get_switchers()
    K = args.k
    num_max_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    window_size = args.window_size
    save_dir = args.save_dir
    deep_encoder = args.deep_encoder == 1
    decoder_init = args.decoder_init
    optimizer = args.optimizer
    loss = args.loss
    mask_frac = args.mask_frac
    save_every = args.save_every
    batch_norm = args.batchnorm
    l2_penalty = args.l2_penalty
    dropout = args.dropout
    seed = args.seed
    display_logs = bool(args.display_logs)
    log_to_wandb = bool(args.wandb_log)
    log.info(f'Job args: {args}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = '{}/{}_{}_init_{}_K_{}_{}_frac_{}_BS_{}_l2_{}_BN_{}_drpt_{}.pt'.format(
                    save_dir,
                    'Deep' if deep_encoder else 'Shallow',
                    window_size,
                    decoder_init,
                    K,
                    loss,
                    mask_frac,
                    batch_size,
                    l2_penalty,
                    batch_norm,
                    dropout
                )
    if not log_to_wandb:
        run_name = None
        log.warn('Run name for wandb not specified. Skipping logging.')
    else:
        run_name = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
        wandb.init(project='neural_admixture',
                   entity='albertdm99',
                   name=run_name,
                   config=args,
                   settings=wandb.Settings(start_method='fork')
                )
        wandb.config.update({'train_samples': len(trX), 'val_samples': len(valX)})
    torch.manual_seed(seed)
    # Initialization
    log.info('Initializing...')
    P_init = switchers['initializations'][decoder_init](trX, K, batch_size, seed)
    ADM = AdmixtureAE(K, trX.shape[1], P_init=P_init, deep_encoder=deep_encoder, batch_norm=bool(batch_norm), lambda_l2=l2_penalty).to(device)
    if log_to_wandb:
        wandb.watch(ADM)
    
    # Optimizer
    optimizer = switchers['optimizers'][optimizer](ADM.parameters(), learning_rate)
    # Losses
    loss_f = switchers['losses'][loss](device, mask_frac)
    loss_weights = None if loss != 'wbce' else torch.tensor(trX.std(axis=0)).float().to(device)

    # Fitting
    log.info('Calling fit...')
    t = Timer()
    t.start()
    actual_num_epochs = ADM.launch_training(trX, optimizer, loss_f, num_max_epochs, device, valX=valX,
                        batch_size=batch_size, loss_weights=loss_weights, display_logs=display_logs,
                        save_every=save_every, save_path=save_path, run_name=run_name)
    elapsed_time = t.stop()
    avg_time_per_epoch = elapsed_time/actual_num_epochs
    if log_to_wandb:
        wandb.run.summary['total_elapsed_time'] = elapsed_time
        wandb.run.summary['avg_epoch_time'] = elapsed_time/actual_num_epochs
    torch.save(ADM.state_dict(), save_path)
    log.info('Fit done.')
    return ADM, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=True, type=float, help='Learning rate')
    parser.add_argument('--batch_size', required=True, type=int, help='Batch size')
    parser.add_argument('--k', required=True, type=int, help='Number of clusters')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument('--decoder_init', required=True, type=str, choices=['random', 'mean_random', 'kmeans', 'kmeans_logit', 'minibatch_kmeans', 'minibatch_kmeans_logit'], help='Decoder initialization')
    parser.add_argument('--loss', required=True, type=str, choices=['mse', 'bce', 'wbce', 'bce_mask', 'mse_mask'], help='Loss function to train')
    parser.add_argument('--mask_frac', required=False, type=float, help='%% of SNPs used in every step (only for masked BCE loss)')
    parser.add_argument('--optimizer', required=True, type=str, choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--save_every', required=True, type=int, help='Save every this number of epochs')
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--window_size', required=True, type=str, help='SNPs window size (e.g. 1K, 50K, 100K)')
    parser.add_argument('--deep_encoder', required=True, type=int, choices=[0, 1], help='Whether to use deep encoder or not')
    parser.add_argument('--batchnorm', required=True, type=int, choices=[0, 1], help='Whether to use batch norm in encoder or not')
    parser.add_argument('--l2_penalty', required=True, type=float, help='L2 penalty on encoder weights')
    parser.add_argument('--display_logs', required=True, type=int, choices=[0, 1], help='Whether to display logs during training or not')
    parser.add_argument('--dropout', required=True, type=float, help='Dropout probability in encoder')
    parser.add_argument('--activation', required=True, type=str, choices=['relu', 'tanh'], help='Activation function for deep encoder layers')
    parser.add_argument('--wandb_log', required=True, type=int, choices=[0, 1], help='Whether to log to wandb or not')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    args = parser.parse_args()
    trX, trY, valX, valY = read_data(args.window_size)
    model, device = fit_model(trX, valX, args)
    if model is None:
        return 1
    if not bool(args.wandb_log):
        return 0
    return plots.generate_plots(model, trX, trY, valX, valY, device, args.batch_size, args.k)

if __name__ == '__main__':
    sys.exit(main())
