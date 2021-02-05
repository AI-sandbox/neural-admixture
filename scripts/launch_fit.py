from admixture_ae import AdmixtureAE
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from weighted_BCE import CustomWBCE
from sklearn.model_selection import train_test_split

def read_data():
    print('Reading data...')
    data = np.load('../data/all_chm_combined_snps_world_2M_with_labels.npz', allow_pickle=True)['snps'][:,::4]
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_save_path = '../outputs/K_{}_lambdal0_{}_CBCE_BS_{}epoch{}.pt'.format(
                    K,
                    lambda_l0,
                    batchsize_samples,
                    num_max_epochs
                )
    print('[INFO] Initializing...')
    if args.decoder_init == 'mean_random':
        X_mean = torch.tensor(dataset.mean(axis=0)).unsqueeze(1)
        P_init = (torch.bernoulli(X_mean.repeat(1, K))-0.5).T.float()
    elif args.decoder_init == 'random':
        P_init = None
    ADM = AdmixtureAE(K, dataset.shape[1], lambda_l0=0, P_init=P_init).to(device)
    if args.optimizer == 'adam':
        optimizer = optim.Adam(ADM.parameters(), lr=learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(ADM.parameters(), lr=learning_rate)
    if args.weight_loss > 0:
        loss_f = CustomWBCE()
        loss_weights = torch.tensor(dataset.std(axis=0)).float().to(device)
    else:
        loss_f = nn.BCELoss()
        loss_weights = None
    print('[INFO] Calling fit...')
    ADM.train(dataset, optimizer, loss_f, num_max_epochs, device, batch_size=batchsize_samples, loss_weights=loss_weights)
    torch.save(ADM.state_dict(), final_save_path)
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
    parser.add_argument('--decoder_init', required=True, type=str, choices=['random', 'mean_random'], help='Decoder initialization')
    parser.add_argument('--weight_loss', required=True, type=int, help='Weight loss per SNP variance')
    parser.add_argument('--optimizer', required=True, type=str, choices=['adam', 'sgd'], help='Optimizer')
    args = parser.parse_args()
    dataset = read_data()
    return fit_model(dataset, args)

if __name__ == '__main__':
    sys.exit(main())
