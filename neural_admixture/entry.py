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
    
    assert not (num_gpus > 1 and not use_multi_gpu), (
        f'Invalid GPU configuration:\n'
        f'- You requested {num_gpus} GPUs\n'
        f'- But the --multi_gpu flag is not set\n'
        f'To use multiple GPUs ({num_gpus}), please add the --multi_gpu flag to your command.'
    )
    
    if sys.argv[1] == 'train':
        from .src import train

        if num_gpus > 1 and use_multi_gpu:
            log.info('Entering multi-GPU training...')
            mp.spawn(train.main, args=(arg_list[2:], num_gpus), nprocs=num_gpus)
        else:
            log.info('Entering single-GPU or CPU training...')
            sys.exit(train.main(0, arg_list[2:], num_gpus))
    
    elif sys.argv[1] == 'infer':
        from .src import inference
        log.info('Entering inference...')
        sys.exit(inference.main(arg_list[2:]))
    
    else:
        log.error(f'Invalid argument {arg_list[1]}. Please run either "neural-admixture train" or "neural-admixture infer"')
        sys.exit(1)