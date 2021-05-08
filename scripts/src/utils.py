import argparse
import h5py
import logging
import os
import wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=True, type=float, help='Learning rate')
    parser.add_argument('--batch_size', required=True, type=int, help='Batch size')
    parser.add_argument('--k', required=True, type=int, help='Number of clusters')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument('--decoder_init', required=True, type=str, choices=['random', 'mean_SNPs', 'mean_random', 'kmeans',
                                                                            'kmeans_logit', 'minibatch_kmeans', 'minibatch_kmeans_logit',
                                                                            'kmeans++', 'binomial', 'pca', 'admixture',
                                                                            'pckmeans', 'supervised'], help='Decoder initialization')
    parser.add_argument('--loss', required=True, type=str, choices=['mse', 'bce', 'wbce', 'bce_mask', 'mse_mask', 'admixture'], help='Loss function to train')
    parser.add_argument('--mask_frac', required=False, type=float, help='%% of SNPs used in every step (only for masked BCE loss)')
    parser.add_argument('--optimizer', required=True, type=str, choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--save_every', required=True, type=int, help='Save every this number of epochs')
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--deep_encoder', required=True, type=int, choices=[0, 1], help='Whether to use deep encoder or not')
    parser.add_argument('--batchnorm', required=True, type=int, choices=[0, 1], help='Whether to use batch norm in encoder or not')
    parser.add_argument('--l2_penalty', required=True, type=float, help='L2 penalty on encoder weights')
    parser.add_argument('--display_logs', required=True, type=int, choices=[0, 1], help='Whether to display logs during training or not')
    parser.add_argument('--dropout', required=True, type=float, help='Dropout probability in encoder')
    parser.add_argument('--activation', required=True, type=str, choices=['relu', 'tanh'], help='Activation function for deep encoder layers')
    parser.add_argument('--wandb_log', required=True, type=int, choices=[0, 1], help='Whether to log to wandb or not')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    parser.add_argument('--multihead', required=False, type=int, default=0, choices=[0,1], help='Whether to train multihead admixture')
    parser.add_argument('--min_k', required=False, type=int, default=3, choices=range(3,15), help='Minimum number of clusters for multihead admixture')
    parser.add_argument('--max_k', required=False, type=int, default=10, choices=range(4,16), help='Maximum number of clusters for multihead admixture')
    parser.add_argument('--chr', required=True, type=str, choices=['1', '22', 'dogs'], help='Chromosome number to train on')
    parser.add_argument('--shuffle', required=True, type=int, choices=[0, 1], help='Whether to shuffle the training data at every epoch')
    parser.add_argument('--pooling', required=False, default=1, type=int, choices=range(1,11), help='Downsample fraction')
    parser.add_argument('--hidden_size', required=False, default=512, type=int, help='Hidden size in encoder and non-linear decoder')
    parser.add_argument('--linear', required=True, type=int, choices=[0, 1], help='Whether to use a linear decoder or not')
    parser.add_argument('--freeze_decoder', required=True, type=int, choices=[0, 1], help='Whether to freeze linear decoder weights')
    parser.add_argument('--init_path', required=False, type=str, help='Path containing precomputed initialization weights to load from')
    parser.add_argument('--supervised', required=True, type=int, choices=[0, 1], help='Whether to use the supervised version or not')
    return parser.parse_args()

def initialize_wandb(should_init, trX, valX, args, out_path, silent=True):
    if not should_init:
        log.warn('Run name for wandb not specified. Skipping logging.')
        return None
    if silent:
        os.environ['WANDB_SILENT'] = 'true'
    run_name = out_path.split('/')[-1][:-3]
    wandb.init(project='neural_admixture',
                entity='albertdm99',
                name=run_name,
                config=args,
                settings=wandb.Settings(start_method='fork')
            )
    wandb.config.update({'train_samples': trX.shape[0],
                         'val_samples': valX.shape[0],
                         'SNPs': trX.shape[1],
                         'out_path': out_path,
                         'averaged_parents': True,
                         'sum_parents': False})
    return run_name

def read_data(chromosome):
    log.info(f'Using data from chromosome {chromosome}')
    f_tr = h5py.File(f'/mnt/gpid08/users/albert.dominguez/data/chr{chromosome}/windowed/train_2gen_avg.h5', 'r')
    f_val = h5py.File(f'/mnt/gpid08/users/albert.dominguez/data/chr{chromosome}/windowed/valid_2gen_avg.h5', 'r')
    return f_tr['snps'], f_tr['populations'], f_val['snps'], f_val['populations']
