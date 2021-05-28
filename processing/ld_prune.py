import allel
import h5py
import logging
import pickle
import sys
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_prune():
    tr_data = h5py.File('/mnt/gpid08/users/albert.dominguez/data/chr1/windowed/train_2gen.h5', 'r')
    trX = tr_data['snps']
    log.info('Locating unlinked variants...')
    t0 = time.time()
    idxs = allel.locate_unlinked(trX[:,:].T, threshold=0.05, step=10, size=100)
    te = time.time()
    log.info('Located {} unlinked variants.\nTime ellapsed: {}'.format(sum(idxs), te-t0))
    log.info('Dumping indices to file...')
    with open('/mnt/gpid08/users/albert.dominguez/data/chr1/unlinked_indices_2gen_0.05.pkl', 'wb') as fb:
        pickle.dump(idxs, fb)
    log.info('Dumped successfully. Exiting...')
    return 0

if __name__ == '__main__':
    sys.exit(run_prune())
