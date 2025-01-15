import logging
import sys
import torch
import torch.multiprocessing as mp
from ._version import __version__

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    log.info(f"Neural ADMIXTURE - Version {__version__}")
    arg_list = tuple(sys.argv)
    assert len(arg_list) > 1, 'Please provide either the argument "train" or "infer" to choose running mode.'
    
    use_multi_gpu = '--multi_gpu' in arg_list
    num_gpus = 0
    if '--num_gpus' in arg_list:
        num_gpus_index = arg_list.index('--num_gpus') + 1
        if num_gpus_index < len(arg_list):
            num_gpus = int(arg_list[num_gpus_index])

    max_devices = torch.cuda.device_count()
    if num_gpus > max_devices:
        log.warning(f"Requested {num_gpus} GPUs, but only {max_devices} are available. Using {max_devices} GPUs.")
        num_gpus = max_devices
        
    if sys.argv[1] == 'train':
        from .src import train

        if num_gpus > 1 and use_multi_gpu:
            mp.spawn(train.main, args=(arg_list[2:], num_gpus), nprocs=num_gpus)
        else:
            sys.exit(train.main(0, arg_list[2:], num_gpus))
    
    elif sys.argv[1] == 'infer':
        from .src import inference
        sys.exit(inference.main(arg_list[2:]))
    
    else:
        log.error(f'Invalid argument {arg_list[1]}. Please run either "neural-admixture train" or "neural-admixture infer"')
        sys.exit(1)