from importlib.metadata import version
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    nadm_version = version("neural-admixture")
    log.info(f"Neural ADMIXTURE - Version {nadm_version}")
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
