import logging
import sys
from ._version import __version__

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    log.info(f"Neural ADMIXTURE - Version {__version__}")
    log.info("[CHANGELOG] Mean imputation for missing data was added in version 1.4.0. To reproduce old behaviour, please use `--imputation zero` when invoking the software.")
    log.info("[CHANGELOG] Default P initialization was changed to 'pckmeans' in version 1.3.0.")
    log.info("[CHANGELOG] Warmup training for initialization of Q was added in version 1.3.0 to improve training stability (only for `pckmeans`).")
    log.info("[CHANGELOG] Convergence check changed so it is performed after 15 epochs in version 1.3.0 to improve training stability.")
    log.info("[CHANGELOG] Default learning rate was changed to 1e-5 instead of 1e-4 in version 1.3.0 to improve training stability.")
    arg_list = tuple(sys.argv)
    assert len(arg_list) > 1, 'Please provide either the argument "train" or "infer" to choose running mode.'
    if sys.argv[1] == 'train':
        from .src import train
        sys.exit(train.main(arg_list[2:]))
    if sys.argv[1] == 'infer':
        from .src import inference
        sys.exit(inference.main(arg_list[2:]))
    log.error(f'Invalid argument {arg_list[1]}. Please run either "neural-admixture train" or "neural-admixture infer"')
    sys.exit(1)
