import logging
import matplotlib.pyplot as plt
import sys
import numpy as np
import plots
import torch
import utils
import wandb
from admixture_ae import AdmixtureAE
from multihead_admixture import AdmixtureMultiHead
from codetiming import Timer
from switchers import Switchers

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
    activation_str = args.activation
    dropout = args.dropout
    multihead = args.multihead
    Ks = [i for i in range(args.min_k, args.max_k+1)]
    assert dropout >= 0 and dropout <= 1
    seed = args.seed
    display_logs = bool(args.display_logs)
    log_to_wandb = bool(args.wandb_log)
    log.info(f'Job args: {args}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = '{}/{}_{}_init_{}_K_{}_{}_frac_{}_BS_{}_l2_{}_BN_{}_drpt_{}_{}_multihead_{}.pt'.format(
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
                    dropout,
                    activation_str,
                    multihead
                )
    
    run_name = utils.initialize_wandb(log_to_wandb, trX, valX, args)
    torch.manual_seed(seed)
    # Initialization
    log.info('Initializing...')
    P_init = switchers['initializations'][decoder_init](trX, Ks if multihead else K, batch_size, seed)
    activation = switchers['activations'][activation_str](0)
    if not multihead:
        model = AdmixtureAE(K, trX.shape[1], P_init=P_init,
                        deep_encoder=deep_encoder, batch_norm=bool(batch_norm),
                        lambda_l2=l2_penalty, encoder_activation=activation,
                        dropout=dropout).to(device)
    else:
        model = AdmixtureMultiHead(Ks, trX.shape[1], P_init=P_init,
                                   batch_norm=bool(batch_norm),
                                   lambda_l2=l2_penalty,
                                   encoder_activation=activation,
                                   dropout=dropout).to(device)
    if log_to_wandb:
        wandb.watch(model)
    
    # Optimizer
    optimizer = switchers['optimizers'][optimizer](model.parameters(), learning_rate)
    # Losses
    loss_f = switchers['losses'][loss](device, mask_frac)
    loss_weights = None if loss != 'wbce' else torch.tensor(trX.std(axis=0)).float().to(device)

    # Fitting
    log.info('Calling fit...')
    t = Timer()
    t.start()
    actual_num_epochs = model.launch_training(trX, optimizer, loss_f, num_max_epochs, device, valX=valX,
                        batch_size=batch_size, loss_weights=loss_weights, display_logs=display_logs,
                        save_every=save_every, save_path=save_path, run_name=run_name)
    elapsed_time = t.stop()
    avg_time_per_epoch = elapsed_time/actual_num_epochs
    if log_to_wandb:
        wandb.run.summary['total_elapsed_time'] = elapsed_time
        wandb.run.summary['avg_epoch_time'] = elapsed_time/actual_num_epochs
    torch.save(model.state_dict(), save_path)
    log.info('Fit done.')
    return model, device

def main():
    args = utils.parse_args()
    trX, trY, valX, valY = utils.read_data(args.window_size)
    model, device = fit_model(trX, valX, args)
    if model is None:
        return 1
    if not bool(args.wandb_log):
        return 0
    return plots.generate_plots(model, trX, trY, valX, valY, device,
                                args.batch_size, k=args.k,
                                is_multihead=args.multihead,
                                min_k=args.min_k, max_k=args.max_k)

if __name__ == '__main__':
    sys.exit(main())
