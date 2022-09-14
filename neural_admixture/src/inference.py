import json
import logging
import sys
import torch
from typing import List

from . import utils
from ..model.neural_admixture import NeuralAdmixture

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main(argv: List[str]):
    """Inference entry point
    """
    args = utils.parse_infer_args(argv)
    log.info('Will use GPU' if torch.cuda.is_available() else 'No GPUs available.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_file_str = args.data_path
    out_name = args.out_name
    weights_file_str = f'{args.save_dir}/{args.name}.pt'
    config_file_str = f'{args.save_dir}/{args.name}_config.json'
    try:
        with open(config_file_str, 'r') as fb:
            config = json.load(fb)
    except FileNotFoundError as fnfe:
        log.error(f'Config file ({config_file_str}) not found. Make sure it is in the correct directory and with the correct name.')
        return 1
    except Exception as e:
        raise e
    log.info('Model config file loaded. Loading weights...')
    model = NeuralAdmixture(config['Ks'], num_features=config['num_snps'])
    model.load_state_dict(torch.load(weights_file_str, map_location=device), strict=True)
    model.to(device)
    log.info('Model weights loaded.')
    X, _, _, _ = utils.read_data(data_file_str)
    assert X.shape[1] == config['num_snps'], 'Number of SNPs in data does not correspond to number of SNPs the network was trained on.'
    log.info('Data loaded and validated. Running inference...')
    _ = utils.get_model_predictions(model, X, bsize=args.batch_size, device=device)
    log.info('Inference run successfully. Writing outputs...')
    utils.write_outputs(model, X, valX=None, bsize=args.batch_size,
                        device=device, run_name=out_name,
                        out_path=args.save_dir, only_Q=True)
    log.info('Exiting...')
    logging.shutdown()
    return 0
