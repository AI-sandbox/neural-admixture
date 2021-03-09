import logging
import numpy as np
import os
import pandas as pd
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DATA_PATH = '/mnt/gpid08/users/albert.dominguez/data/chr22'

def main():
    for ancestry in ['AFR', 'AMR', 'EAS', 'EUR', 'OCE', 'SAS', 'WAS']:
        log.info(f'Splitting founders of {ancestry}')
        df = pd.read_csv(os.path.join(DATA_PATH, f'all/{ancestry}.map'),delimiter="\t",header=None,comment="#",dtype=str)
        train, valid, test = np.split(df.sample(frac=1, random_state=42), [int(.7*len(df)), int(.85*len(df))])
        train.to_csv(os.path.join(DATA_PATH, f'train/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
        valid.to_csv(os.path.join(DATA_PATH, f'valid/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
        test.to_csv(os.path.join(DATA_PATH, f'test/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
    return 0

if __name__ == '__main__':
    sys.exit(main())
