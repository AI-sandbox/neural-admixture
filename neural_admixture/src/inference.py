import json
import logging
import sys
import os
import torch
from typing import List
from pathlib import Path

from . import utils
from .loaders import dataloader_inference
from ..model.neural_admixture import Q_P
from ..model.switchers import Switchers
from ..model.IPCA_GPU import IncrementalPCAonGPU
torch.serialization.add_safe_globals([IncrementalPCAonGPU])

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main(argv: List[str]):
    """Inference entry point
    """
    # CHECK WHETHER THERE'S GPU OR NOT:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        num_gpus = 1
        pin_non_blocking = False
    else:
        device = torch.device('cpu')
        num_gpus = 0
        pin_non_blocking = True
        
    utils.print_neural_admixture_banner()
    log.info(f"There are {os.cpu_count()} CPUs and {num_gpus} GPUs available.")
    
    # LOAD ARGUMENTS:
    args = utils.parse_infer_args(argv)

    data_file_str = args.data_path
    out_name = args.out_name
    weights_file_str = f'{args.save_dir}/{args.name}.pt'
    config_file_str = f'{args.save_dir}/{args.name}_config.json'
    init_file = f'{args.name}.pkl'
    pca_file_str = f'{Path(args.save_dir) / init_file}'
    seed = int(args.seed)
    batch_size_inference_Q = int(args.batch_size)
    generator = torch.Generator().manual_seed(seed)
    
    # LOAD MODEL:
    try:
        with open(config_file_str, 'r') as fb:
            config = json.load(fb)
    except FileNotFoundError as fnfe:
        log.error(f'Config file ({config_file_str}) not found. Make sure it is in the correct directory and with the correct name.')
        return 1
    except Exception as e:
        raise e
    log.info('Model config file loaded. Loading weights...')
    
    switchers = Switchers.get_switchers()
    activation = switchers['activations'][config['activation']](0)
    model = Q_P(int(config['hidden_size']), int(config['num_features']), int(config['k']), activation)
    model.load_state_dict(torch.load(weights_file_str, weights_only=True, map_location=device))
    model.to(device)
    log.info('Model weights loaded.')
    
    # LOAD DATA:
    trX = utils.read_data(data_file_str, master=True)
    data = utils.initialize_data(True, trX)
    
    # LOAD PCA:
    if os.path.exists(pca_file_str):
        pca_obj = torch.load(pca_file_str, weights_only=True, map_location=device)
        pca_obj.to(device)
        log.info('PCA loaded.')
        X_pca = pca_obj.transform(data)
        assert pca_obj.n_features_in_ == data.shape[1], 'Computed PCA and training data do not have the same number of features'
    else:
        raise FileNotFoundError
    input = X_pca.to(device)
    
    # INFERENCE:
    model.eval()
    Q = torch.tensor([], device=device)
    
    with torch.inference_mode():
        dataloader = dataloader_inference(input, batch_size_inference_Q, seed, generator, num_gpus, pin=pin_non_blocking)
        for input_step in dataloader:
            input_step = input_step.to(device)
            out = model(input_step, 'P2', only_probs=True)
            Q = torch.cat((Q, out), dim=0)
    log.info('Inference run successfully. Writing outputs...')
    
    # WRITE OUTPUTS:
    utils.write_outputs(Q.cpu().numpy(), out_name, int(config['k']), args.save_dir)
    log.info('Exiting...')
    logging.shutdown()
    return

if __name__ == '__main__':
    main(sys.argv[1:])
