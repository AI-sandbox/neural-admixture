from admixture_ae import AdmixtureAE
import argparse
import h5py
import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import plots
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from custom_losses import MaskedBCE, MaskedMSE, WeightedBCE
from codetiming import Timer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

os.environ['WANDB_SILENT'] = 'true'

def read_data(window_size=0):
    log.info('Reading data...')
    f_tr = h5py.File(f'/home/usuaris/imatge/albert.dominguez/neural-admixture/data/chr22/prepared/train{window_size}.h5', 'r')
    f_val = h5py.File(f'/home/usuaris/imatge/albert.dominguez/neural-admixture/data/chr22/prepared/valid{window_size}.h5', 'r')
    return f_tr['snps'], f_tr['populations'], f_val['snps'], f_val['populations']


def fit_model(trX, valX, args):
    K = args.k
    gamma_l0 = args.gamma_l0
    lambda_l0 = args.lambda_l0
    num_max_epochs = args.epochs
    batch_size = args.bs
    learning_rate = args.lr
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
        run_name = save_path.split('/')[-1][:-3]
        wandb.init(project='neural_admixture', entity='albertdm99', name=run_name, config=args)
        wandb.config.update({'train_samples': len(trX), 'val_samples': len(valX)})
    log.info('Initializing...')
    if decoder_init == 'mean_random':
        X_mean = torch.tensor(np.mean(trX, axis=0)).unsqueeze(1)
        P_init = (torch.bernoulli(X_mean.repeat(1, K))-0.5).T.float()
        del X_mean
    elif decoder_init == 'random':
        P_init = None
    elif decoder_init.startswith('kmeans'):
        log.info('Getting k-Means cluster centroids...')
        k_means_obj = KMeans(n_clusters=K, random_state=42).fit(trX)
        if decoder_init.endswith('logit'):
            P_init = torch.clamp(torch.tensor(k_means_obj.cluster_centers_).float(), min=1e-4, max=1-1e-4)
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                return None, None
        else:
            P_init = torch.tensor(k_means_obj.cluster_centers_).float()
        del k_means_obj
    elif decoder_init.startswith('minibatch_kmeans'):
        log.info('Getting minibatch k-Means cluster centroids...')
        k_means_obj = MiniBatchKMeans(n_clusters=K, batch_size=batch_size, random_state=42).fit(trX)
        if decoder_init.endswith('logit'):
            P_init = torch.clamp(torch.tensor(k_means_obj.cluster_centers_).float(), min=1e-4, max=1-1e-4)
            P_init = torch.logit(P_init, eps=1e-4)
            if sum(torch.isnan(P_init.flatten())).item() > 0:
                log.error('Initialization weights contain NaN values.')
                return None, None
        else:
            P_init = torch.tensor(k_means_obj.cluster_centers_).float()
        del k_means_obj
    ADM = AdmixtureAE(K, trX.shape[1], lambda_l0=0, P_init=P_init, deep_encoder=deep_encoder, batch_norm=bool(batch_norm), lambda_l2=l2_penalty).to(device)
    if log_to_wandb:
        wandb.watch(ADM)
    if optimizer == 'adam':
        optimizer = optim.Adam(ADM.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(ADM.parameters(), lr=learning_rate)
    loss_weights = None
    if loss == 'mse':
        loss_f = nn.MSELoss()
    elif loss == 'bce':
        loss_f = nn.BCELoss()
    elif loss == 'wbce':
        loss_f = WeightedBCE()
        loss_weights = torch.tensor(trX.std(axis=0)).float().to(device)
    elif loss == 'bce_mask':
        loss_f = MaskedBCE(device, mask_frac=mask_frac)
    elif loss == 'mse_mask':
        loss_f = MaskedMSE(device, mask_frac=mask_frac)
    log.info('Calling fit...')
    t = Timer()
    t.start()
    actual_num_epochs = ADM.launch_training(trX, optimizer, loss_f, num_max_epochs, device, valX=valX,
                        batch_size=batch_size, loss_weights=loss_weights, display_logs=display_logs,
                        save_every=save_every, save_path=save_path, run_name=run_name)
    elapsed_time = t.stop()
    avg_time_per_epoch = elapsed_time/actual_num_epochs
    wandb.run.summary['total_elapsed_time'] = elapsed_time
    wandb.run.summary['avg_epoch_time'] = elapsed_time/actual_num_epochs
    torch.save(ADM.state_dict(), save_path)
    log.info('Fit done.')
    return ADM, device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', required=True, type=float, help='Learning rate')
    parser.add_argument('--bs', required=True, type=int, help='Batch size')
    parser.add_argument('--k', required=True, type=int, help='Number of clusters')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument('--lambda_l0', required=True, type=float, help='L0 Lambda parameter')
    parser.add_argument('--gamma_l0', required=False, type=float, default=0.01, help='L0 Gamma parameter')
    parser.add_argument('--beta_l0', required=False, type=float, default=0.01, help='L0 Beta parameter')
    parser.add_argument('--theta_l0', required=False, type=float, default=0.01, help='L0 Theta parameter')
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
    parser.add_argument('--wandb_log', required=True, type=int, choices=[0, 1], help='Whether to log to wandb or not')
    args = parser.parse_args()
    trX, trY, valX, valY = read_data(args.window_size)
    model, device = fit_model(trX, valX, args)
    if model is None:
        return 1
    if not bool(args.wandb_log):
        return 0
    return plots.generate_plots(model, trX, trY, valX, valY, device, args.bs, args.k)

if __name__ == '__main__':
    sys.exit(main())
