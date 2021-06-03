import argparse
import h5py
import logging
import os
import wandb

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', required=False, default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--batch_size', required=False, default=200, type=int, help='Batch size')
    parser.add_argument('--epochs', required=True, type=int, help='Number of epochs')
    parser.add_argument('--decoder_init', required=True, type=str, choices=['random', 'mean_SNPs', 'mean_random', 'kmeans',
                                                                            'minibatch_kmeans', 'kmeans++', 'binomial',
                                                                            'pca', 'admixture', 'pckmeans', 'supervised'], help='Decoder initialization')
    parser.add_argument('--optimizer', required=False, default='adam', type=str, choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--save_every', required=False, default=50, type=int, help='Save every this number of epochs')
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--l2_penalty', required=False, default=0.01, type=float, help='L2 penalty on encoder weights')
    parser.add_argument('--display_logs', required=True, type=int, choices=[0, 1], help='Whether to display logs during training or not')
    parser.add_argument('--activation', required=False, default='relu', type=str, choices=['relu', 'tanh'], help='Activation function for deep encoder layers')
    parser.add_argument('--wandb_log', required=False, default=0, type=int, choices=[0, 1], help='Whether to log to wandb or not')
    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    parser.add_argument('--min_k', required=False, type=int, default=7, help='Minimum number of clusters for multihead admixture')
    parser.add_argument('--max_k', required=False, type=int, default=7, help='Maximum number of clusters for multihead admixture')
    parser.add_argument('--chr', required=False, type=str, choices=['1', '22'], help='Chromosome number to train on')
    parser.add_argument('--shuffle', required=False, default=1, type=int, choices=[0, 1], help='Whether to shuffle the training data at every epoch')
    parser.add_argument('--hidden_size', required=False, default=512, type=int, help='Hidden size in encoder and non-linear decoder')
    parser.add_argument('--linear', required=False, default=True, type=int, choices=[0, 1], help='Whether to use a linear decoder or not')
    parser.add_argument('--freeze_decoder', required=True, type=int, choices=[0, 1], help='Whether to freeze linear decoder weights')
    parser.add_argument('--init_path', required=False, type=str, help='Path containing precomputed initialization weights to load from')
    parser.add_argument('--supervised', required=True, type=int, choices=[0, 1], help='Whether to use the supervised version or not')
    parser.add_argument('--plot_k', required=False, default=7, type=int, help='Value of K used for the post-training plots')
    parser.add_argument('--name', required=False, type=str, help='Experiment name (will be used for output filenames)')
    parser.add_argument('--dataset', required=False, type=str, help='Dataset to be used')
    parser.add_argument('--data_path', required=False, type=str, help='Path containing the data')
    parser.add_argument('--wandb_user', required=False, type=str, help='wandb user')
    parser.add_argument('--wandb_project', required=False, type=str, help='wandb project')
    parser.add_argument('--pca_path', required=False, type=str, help='Path containing PCA object, used for plots')
    return parser.parse_args()

def initialize_wandb(should_init, trX, valX, args, out_path, silent=True):
    if not should_init:
        log.warn('Run name for wandb not specified. Skipping logging.')
        return None
    if silent:
        os.environ['WANDB_SILENT'] = 'true'
    run_name = out_path.split('/')[-1][:-3]
    wandb.init(project=args.wandb_project,
               entity=args.wandb_user,
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

def read_data(tr_file, val_file):
    f_tr = h5py.File(tr_file, 'r')
    f_val = h5py.File(val_file, 'r')
    return f_tr['snps'], f_tr['populations'], f_val['snps'], f_val['populations']
