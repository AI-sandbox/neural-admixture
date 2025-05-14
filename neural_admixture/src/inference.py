import json
import logging
import sys
import os
import torch
import time

from typing import List

from . import utils
from .loaders import dataloader_admixture
from ..model.neural_admixture import Q_P

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def main(argv: List[str]):
    """Inference entry point
    """
    # CHECK WHETHER THERE'S GPU OR NOT:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        num_gpus = 1
        pin = False
    else:
        device = torch.device('cpu')
        num_gpus = 0
        pin = True
        
    # LOAD ARGUMENTS:
    args = utils.parse_infer_args(argv)
    log.info(f"    There are {args.num_cpus} CPUs and {num_gpus} GPUs available.")
    log.info("")
    data_file_str = args.data_path
    out_name = args.out_name
    model_file_str = f'{args.save_dir}/{args.name}.pt'
    config_file_str = f'{args.save_dir}/{args.name}_config.json'
    seed = int(args.seed)
    batch_size_inference_Q = int(args.batch_size)
    num_cpus = args.num_cpus
    generator = torch.Generator().manual_seed(seed)
    
    # LOAD MODEL:
    try:
        with open(config_file_str, 'r') as fb:
            config = json.load(fb)
    except FileNotFoundError as fnfe:
        log.error(f"    Config file ({config_file_str}) not found. Make sure it is in the correct directory and with the correct name.")
        return 1
    except Exception as e:
        raise e
    log.info("    Model config file loaded. Loading weights...")
    
    model = Q_P(int(config['hidden_size']), int(config['num_features']), int(config['k']), activation, is_train=False)
    model.load_state_dict(torch.load(model_file_str, weights_only=True, map_location=device))
    model.to(device)
    log.info("    Model weights loaded.")
    log.info("")
    
    # LOAD DATA:
    t0 = time.time()
    trX, _ = utils.read_data(data_file_str, master=True)
    data, _ = utils.initialize_data(True, trX)
    
    # INFERENCE:
    model.eval()
    Q = torch.tensor([], device=device)
    log.info("    Running inference...")
    with torch.inference_mode():
        dataloader = dataloader_admixture(data, batch_size_inference_Q, seed, generator, num_gpus, pin, num_cpus)
        for input_step in dataloader:
            input_step = input_step.to(device)
            probs = model(input_step)
            Q = torch.cat((Q, probs), dim=0)
    log.info("    Inference run successfully! Writing outputs...!")
    
    # WRITE OUTPUTS:
    utils.write_outputs(Q.cpu().numpy(), out_name, int(config['k']), args.save_dir)
    
    t1 = time.time()
    log.info("")
    log.info(f"    Total elapsed time: {t1-t0:.2f} seconds.")
    log.info("")
    
    logging.shutdown()

if __name__ == '__main__':
    main(sys.argv[1:])
