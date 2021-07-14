import logging
import matplotlib.pyplot as plt
import numpy as np
import pickle
import plots
import sys
import torch
import torch.nn as nn
import utils
import wandb
sys.path.append('..')
from model.neural_admixture import NeuralAdmixture
from codetiming import Timer
from parallel import CustomDataParallel
from pathlib import Path
from switchers import Switchers

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def fit_model(trX, args, valX=None, trY=None, valY=None):
    switchers = Switchers.get_switchers()
    num_max_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    save_dir = args.save_dir
    decoder_init = args.decoder_init
    optimizer_str = args.optimizer
    save_every = args.save_every
    l2_penalty = args.l2_penalty
    activation_str = args.activation
    shuffle = args.shuffle
    hidden_size = args.hidden_size
    linear = bool(args.linear)
    freeze_decoder = bool(args.freeze_decoder)
    init_path = args.init_path
    supervised = bool(args.supervised)
    name = args.name
    Ks = [i for i in range(args.min_k, args.max_k+1)]
    assert not (supervised and len(Ks) != 1), 'Supervised version is currently only available on a single head'
    seed = args.seed
    display_logs = bool(args.display_logs)
    log_to_wandb = bool(args.wandb_log)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log.info(f'Job args: {args}')
    log.info('Using {} GPU(s)'.format(torch.cuda.device_count()) if torch.cuda.is_available() else 'No GPUs available.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_name = name
    if log_to_wandb:
        utils.initialize_wandb(run_name, trX, valX, args, save_dir)
    save_path = '{}/{}.pt'.format(save_dir, run_name)
    torch.manual_seed(seed)
    # Initialization
    log.info('Initializing...')
    if linear:
        P_init = switchers['initializations'][decoder_init](trX, trY, Ks, batch_size, seed, init_path)
    else:
        P_init = None
        log.info('Non-linear decoder weights will be randomly initialized.')
    activation = switchers['activations'][activation_str](0)
    log.info('Variants: {}'.format(trX.shape[1]))
    model = NeuralAdmixture(Ks, trX.shape[1], P_init=P_init,
                                lambda_l2=l2_penalty,
                                encoder_activation=activation,
                                linear=linear, hidden_size=hidden_size,
                                freeze_decoder=freeze_decoder,
                                supervised=supervised)
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
    model.to(device)
    if log_to_wandb:
        wandb.watch(model)
    
    # Optimizer
    optimizer = switchers['optimizers'][optimizer_str](filter(lambda p: p.requires_grad, model.parameters()), learning_rate)
    log.info('Optimizer successfully loaded.')

    loss_f = nn.BCELoss(reduction='mean')
    log.info('Going to train {} head{}.'.format(len(Ks), 's' if len(Ks) > 1 else ''))
    # Fitting
    log.info('Calling fit...')
    t = Timer()
    t.start()
    actual_num_epochs = model.launch_training(trX, optimizer, loss_f, num_max_epochs, device, valX=valX,
                       batch_size=batch_size, display_logs=display_logs, save_every=save_every,
                       save_path=save_path, run_name=run_name, plot_every=0,
                       trY=trY, valY=valY, shuffle=shuffle, seed=seed, log_to_wandb=log_to_wandb)
    elapsed_time = t.stop()
    if log_to_wandb:
        wandb.run.summary['total_elapsed_time'] = elapsed_time
        wandb.run.summary['avg_epoch_time'] = elapsed_time/actual_num_epochs
    torch.save(model.state_dict(), save_path)
    log.info('Fit done.')
    return model, device

def main():
    args = utils.parse_args()
    switchers = Switchers.get_switchers()
    if args.dataset is not None:
        tr_file, val_file = switchers['data'][args.dataset](args.data_path)
        tr_pops_f, val_pops_f = '', ''
    else:
        tr_file, val_file = args.data_path, args.validation_data_path
        tr_pops_f, val_pops_f = args.populations_path, args.validation_populations_path

    trX, trY, valX, valY = utils.read_data(tr_file, val_file, tr_pops_f, val_pops_f)
    model, device = fit_model(trX, args, valX, trY, valY)
    utils.write_outputs(model, trX, valX, args.batch_size, device, args.name, args.save_dir)
    return 0

if __name__ == '__main__':
    sys.exit(main())
