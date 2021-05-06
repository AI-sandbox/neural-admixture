import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plots
import sys
import torch
import utils
import uuid
import wandb
from admixture_ae import AdmixtureAE
from multihead_admixture import AdmixtureMultiHead
from codetiming import Timer
from parallel import CustomDataParallel
from switchers import Switchers

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def fit_model(trX, valX, args, trY=None, valY=None):
    switchers = Switchers.get_switchers()
    K = args.k
    num_max_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    save_dir = args.save_dir
    deep_encoder = args.deep_encoder == 1
    decoder_init = args.decoder_init
    optimizer_str = args.optimizer
    loss = args.loss
    mask_frac = args.mask_frac
    save_every = args.save_every
    batch_norm = args.batchnorm
    l2_penalty = args.l2_penalty
    activation_str = args.activation
    dropout = args.dropout
    multihead = args.multihead
    shuffle = args.shuffle
    pooling = args.pooling
    alternate = bool(args.alternate)
    hidden_size = args.hidden_size
    linear = bool(args.linear)
    freeze_decoder = bool(args.freeze_decoder)
    init_path = args.init_path
    supervised = bool(args.supervised)
    Ks = [i for i in range(args.min_k, args.max_k+1)]
    assert not supervised or (len(Ks) == 1 or not multihead), 'Supervised version is only available on a single head'
    assert dropout >= 0 and dropout <= 1, 'Dropout must be between 0 and 1'
    seed = args.seed
    display_logs = bool(args.display_logs)
    log_to_wandb = bool(args.wandb_log)
    log.info(f'Job args: {args}')
    log.info('Using {} GPU(s)'.format(torch.cuda.device_count()) if torch.cuda.is_available() else 'No GPUs available.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    save_path = '{}/{}.pt'.format(save_dir, uuid.uuid4().hex)
    run_name = utils.initialize_wandb(log_to_wandb, trX, valX, args, save_path)
    torch.manual_seed(seed)
    # Initialization
    log.info('Initializing...')
    if linear:
        P_init = switchers['initializations'][decoder_init](trX, Ks if multihead else K, batch_size, seed, init_path)
    else:
        P_init = None
        log.info('Non-linear decoder weights will be randomly initialized.')
    activation = switchers['activations'][activation_str](0)
    log.info('Features: {}'.format(trX.shape[1]))
    if not multihead:
        model = AdmixtureAE(K, trX.shape[1], P_init=P_init,
                        deep_encoder=deep_encoder, batch_norm=bool(batch_norm),
                        lambda_l2=l2_penalty, encoder_activation=activation,
                        dropout=dropout)
    else:
        model = AdmixtureMultiHead(Ks, trX.shape[1], P_init=P_init,
                                   batch_norm=bool(batch_norm),
                                   lambda_l2=l2_penalty,
                                   encoder_activation=activation,
                                   dropout=dropout, pooling=pooling,
                                   linear=linear, hidden_size=hidden_size,
                                   freeze_decoder=freeze_decoder,
                                   supervised=supervised)
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
    model.to(device)
    if log_to_wandb:
        wandb.watch(model)
    
    # Optimizer
    if alternate:
        optimizer = switchers['optimizers'][optimizer_str]( # Encoder optimizer
            list(model.batch_norm.parameters())+list(model.common_encoder.parameters())+list(model.multihead_encoder.parameters()),
            learning_rate
        )
        optimizer_dec = switchers['optimizers'][optimizer_str]( # Decoder optimizer
            model.decoders.parameters(),
            learning_rate
        )
    else:
        optimizer = switchers['optimizers'][optimizer_str](filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    log.info('Optimizer successfully loaded.')
    # Losses
    loss_f = switchers['losses'][loss](device, mask_frac)
    loss_weights = None if loss != 'wbce' else torch.tensor(trX.std(axis=0)).float().to(device)

    # Fitting
    log.info('Calling fit...')
    t = Timer()
    t.start()
    actual_num_epochs = model.launch_training(trX, optimizer, loss_f, num_max_epochs, device, valX=valX,
                       batch_size=batch_size, loss_weights=loss_weights, display_logs=display_logs,
                       save_every=save_every, save_path=save_path, run_name=run_name, plot_every=0,
                       trY=trY, valY=valY, shuffle=shuffle, seed=seed, optimizer_2=optimizer_dec if alternate else None)
    elapsed_time = t.stop()
    avg_time_per_epoch = elapsed_time/actual_num_epochs
    if log_to_wandb:
        wandb.run.summary['total_elapsed_time'] = elapsed_time
        wandb.run.summary['avg_epoch_time'] = elapsed_time/actual_num_epochs
    torch.save(model.state_dict(), save_path)
    log.info('Fit done.')
    return model, P_init, device

def main():
    args = utils.parse_args()
    trX, trY, valX, valY = utils.read_data(args.chr)
    model, P_init, device = fit_model(trX, valX, args, trY, valY)
    if model is None:
        return 1
    if not bool(args.wandb_log):
        return 0
    pca_path = '/mnt/gpid08/users/albert.dominguez/data/chr{}/pca_gen2_avg.pkl'.format(args.chr)
    try:
        with open(pca_path, 'rb') as fb:
            pca_obj = pickle.load(fb)
    except FileNotFoundError as fnf:
        log.exception(fnf)
        pca_obj = None
        pass
    return plots.generate_plots(model, trX, trY, valX, valY, device,
                                args.batch_size, k=args.k,
                                is_multihead=args.multihead,
                                to_wandb=bool(args.wandb_log),
                                min_k=args.min_k, max_k=args.max_k,
                                P_init=P_init, pca_obj=pca_obj,
                                linear=args.linear)

if __name__ == '__main__':
    sys.exit(main())
