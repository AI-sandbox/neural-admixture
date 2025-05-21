import json
import logging
import sys
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
    log.info(f"    There is {num_gpus} GPUs available.")
    log.info("")
    data_file_str = args.data_path
    out_name = args.out_name
    model_file_str = f'{args.save_dir}/{args.name}.pt'
    config_file_str = f'{args.save_dir}/{args.name}_config.json'
    seed = int(args.seed)
    batch_size_inference_Q = int(args.batch_size)
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
    state_dict = torch.load(model_file_str, map_location=device, weights_only=True)
    V = state_dict.get("V")
    model = Q_P(int(config['hidden_size']), int(config['num_features']), ks_list=config['ks'], V=V, is_train=False)
    model.load_state_dict(state_dict)
    model.to(device)
    log.info("")
    log.info("    Model weights loaded.")
    log.info("")
    
    # LOAD DATA:
    t0 = time.time()
    data, N, M = utils.read_data(data_file_str)
    data = torch.as_tensor(data, dtype=torch.uint8, device=device)
    
    # INFERENCE:
    model.eval()
    Qs = [torch.tensor([], device=device) for _ in config['ks']]
    log.info("    Running inference...")
    with torch.inference_mode():
        dataloader = dataloader_admixture(data, batch_size_inference_Q, num_gpus, seed, generator, y=None,  shuffle=False)
        for x_step, _ in dataloader:
            probs, _ = model(x_step)
            for i, k in enumerate(config['ks']):
                Qs[i]= torch.cat((Qs[i], probs[i]), dim=0)
    log.info("    Inference run successfully! Writing outputs...!")
    
    # WRITE OUTPUTS:
    K = config['ks'][0]
    if len(config['ks'])>1:
        K = config['ks'][0]
        min_k = None
        max_k = None
    else:
        K = None
        min_k = config['ks'][0]
        max_k = config['ks'][-1]
        
    utils.write_outputs(Qs.cpu().numpy(), out_name, K, min_k, max_k, args.save_dir)
    
    t1 = time.time()
    log.info("")
    log.info(f"    Total elapsed time: {t1-t0:.2f} seconds.")
    log.info("")
    
    logging.shutdown()

if __name__ == '__main__':
    main(sys.argv[1:])
