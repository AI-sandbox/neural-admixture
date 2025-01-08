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

    num_devices = torch.cuda.device_count()
    
    if sys.argv[1] == 'train':
        from .src import train

        # FIXME: multi-gpu training should be disabled by default unless explicilty asked for by the user to avoid accidental HW hijacking
        if num_devices > 1:
            mp.spawn(train.main, args=(arg_list[2:],), nprocs=num_devices)
        else:
            sys.exit(train.main(0, arg_list[2:]))
    
    if sys.argv[1] == 'infer':
        from .src import inference
        sys.exit(inference.main(arg_list[2:]))
    
    else:
        log.error(f'Invalid argument {arg_list[1]}. Please run either "neural-admixture train" or "neural-admixture infer"')
        sys.exit(1)