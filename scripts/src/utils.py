import argparse
import h5py
import logging
import os
import wandb
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=True, type=float, help='Learning rate')
    parser.add_argument('--batch_size', required=True, type=int, help='Batch size')
    parser.add_argument('--k', required=True, type=int, help='Number of clusters')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument('--decoder_init', required=True, type=str, choices=['random', 'mean_SNPs', 'mean_random', 'kmeans', 'kmeans_logit', 'minibatch_kmeans', 'minibatch_kmeans_logit'], help='Decoder initialization')
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
    parser.add_argument('--multihead', required=False, type=int, default=0, choices=[0,1], help='Whether to train multihead admixture')
    parser.add_argument('--min_k', required=False, type=int, default=3, choices=range(3,10), help='Minimum number of clusters for multihead admixture')
    parser.add_argument('--max_k', required=False, type=int, default=10, choices=range(4,11), help='Maximum number of clusters for multihead admixture')
    return parser.parse_args()

def initialize_wandb(should_init, trX, valX, args, silent=True):
    if not should_init:
        log.warn('Run name for wandb not specified. Skipping logging.')
        return None
    if silent:
        os.environ['WANDB_SILENT'] = 'true'
    run_name = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    wandb.init(project='neural_admixture',
                entity='albertdm99',
                name=run_name,
                config=args,
                settings=wandb.Settings(start_method='fork')
            )
    wandb.config.update({'train_samples': len(trX), 'val_samples': len(valX)})
    return run_name

def read_data(window_size='0'):
    log.info('Reading data...')
    f_tr = h5py.File(f'/home/usuaris/imatge/albert.dominguez/neural-admixture/data/chr22/prepared/train{window_size}.h5', 'r')
    f_val = h5py.File(f'/home/usuaris/imatge/albert.dominguez/neural-admixture/data/chr22/prepared/valid{window_size}.h5', 'r')
    return f_tr['snps'], f_tr['populations'], f_val['snps'], f_val['populations']