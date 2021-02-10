from admixture_ae import AdmixtureAE
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from custom_losses import WeightedBCE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split

def read_data(window_size=0):
    print('Reading data...')
    if window_size == 0:
        data = np.load('../data/all_chm_combined_snps_world_2M_with_labels.npz', allow_pickle=True)['snps']
    else:
        data = np.load('../data/all_chm_combined_snps_world_2M_with_labels.npz', allow_pickle=True)['snps'][:,:window_size]
    print('Splitting dataset...')
    trX, _, _, _ = train_test_split(data, [0]*len(data), test_size=.15, random_state=42)
    del data
    return trX


def fit_model(dataset, args):
    K = args.k
    gamma_l0 = args.gamma_l0
    lambda_l0 = args.lambda_l0
    num_max_epochs = args.epochs
    batchsize_samples = args.bs
    learning_rate = args.lr
    window_size = args.window_size
    save_dir = args.save_dir
    deep_encoder = args.deep_encoder == 1
    decoder_init = args.decoder_init
    optimizer = args.optimizer
    weight_loss = args.weight_loss
    save_every = args.save_every
    print('[INFO] Job args:', args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = '{}/{}_Windowed_{}_init_{}_K_{}_lambdal0_{}_WBCE_BS_{}.pt'.format(
                    save_dir,
                    'Deep' if deep_encoder else 'Shallow',
                    window_size,
                    decoder_init,
                    K,
                    lambda_l0,
                    batchsize_samples
                )
    print('[INFO] Initializing...')
    if decoder_init == 'mean_random':
        X_mean = torch.tensor(dataset.mean(axis=0)).unsqueeze(1)
        P_init = (torch.bernoulli(X_mean.repeat(1, K))-0.5).T.float()
    elif decoder_init == 'random':
        P_init = None
    elif decoder_init.startswith('kmeans'):
        print('[INFO] Getting k-Means cluster centroids...')
        k_means_obj = KMeans(n_clusters=K, random_state=42).fit(dataset)
        if decoder_init.endswith('logit'):
            centr = torch.tensor(k_means_obj.cluster_centers_).float()
            P_init = torch.log(centr)/(1-centr) # logit
            del centr
        else:
            P_init = torch.tensor(k_means_obj.cluster_centers_).float()
        del k_means_obj
    elif decoder_init.startswith('minibatch_kmeans'):
        print('[INFO] Getting minibatch k-Means cluster centroids...')
        k_means_obj = MiniBatchKMeans(n_clusters=K, batch_size=batchsize_samples, random_state=42).fit(dataset)
        if decoder_init.endswith('logit'):
            P_init = torch.logit(torch.tensor(k_means_obj.cluster_centers_).float(), eps=1e-8)
            del centr
        else:
            P_init = torch.tensor(k_means_obj.cluster_centers_).float()
        del k_means_obj
    ADM = AdmixtureAE(K, dataset.shape[1], lambda_l0=0, P_init=P_init, deep_encoder=deep_encoder).to(device)
    if optimizer == 'adam':
        optimizer = optim.Adam(ADM.parameters(), lr=learning_rate)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(ADM.parameters(), lr=learning_rate)
    if weight_loss > 0:
        loss_f = WeightedBCE()
        loss_weights = torch.tensor(dataset.std(axis=0)).float().to(device)
    else:
        loss_f = nn.MSELoss()
        loss_weights = None
    print('[INFO] Calling fit...')
    ADM.train(dataset, optimizer, loss_f, num_max_epochs, device, batch_size=batchsize_samples, loss_weights=loss_weights, save_every=save_every, save_path=save_path)
    torch.save(ADM.state_dict(), save_path)
    print('[INFO] Fit done.')
    return 0

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
    parser.add_argument('--weight_loss', required=True, type=int, choices=[0, 1], help='Weight loss per SNP variance')
    parser.add_argument('--optimizer', required=True, type=str, choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--save_every', required=True, type=int, help='Save every this number of epochs')
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--window_size', required=True, type=int, help='SNPs window size')
    parser.add_argument('--deep_encoder', required=True, type=int, choices=[0, 1], help='Whether to use deep encoder or not')
    args = parser.parse_args()
    dataset = read_data(args.window_size)
    return fit_model(dataset, args)

if __name__ == '__main__':
    sys.exit(main())
