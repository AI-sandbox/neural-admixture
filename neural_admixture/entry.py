import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    assert len(sys.argv) > 1, 'Please provide either the argument "train" or "infer" to choose running mode.'
    if sys.argv[1] == 'train':
        from src import train
        sys.exit(train.main())
    if sys.argv[1] == 'infer':
        from src import inference
        sys.exit(inference.main())
    log.error(f'Invalid argument {sys.argv[1]}. Please run either "neural-admixture train" or "neural-admixture infer"')
    sys.exit(1)