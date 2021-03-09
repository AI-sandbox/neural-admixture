import gc
import glob
import h5py
import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def get_snps_by_pop_and_set(pop, which_set, window_size=5000):
    data_path = '/mnt/gpid08/users/albert.dominguez/data/chr22'
    print(f'{data_path}/{which_set}/{pop}/gen_*/{pop}_mat_vcf_2d.npy')
    log.info(f'Fetching SNPs for population {pop} and set {which_set}')
    for i, snps_arr in enumerate(glob.glob(f'{data_path}/{which_set}/{pop}/gen_*/mat_vcf_2d.npy')):
        aux = np.load(snps_arr, mmap_mode='r')[:,:window_size]
        if i == 0:
            arr = np.empty((0, aux.shape[1]), int)
        arr = np.vstack((arr, aux))
        del aux
        gc.collect()
    log.info('Done.')
    return arr

def main():
    opath = '/mnt/gpid08/users/albert.dominguez/data/chr22/windowed'
    pops = ['AFR', 'AMR', 'EAS', 'EUR', 'OCE', 'SAS', 'WAS']
    window_size = 317408
    for which_set in ['train', 'valid', 'test']:
        for i in range(1,len(pops)):
            if i == 1:
                pop0, pop1 = get_snps_by_pop_and_set(pops[0], which_set, window_size=window_size), get_snps_by_pop_and_set(pops[1], which_set, window_size=window_size)
                X = np.vstack((pop0, pop1))
                y = np.concatenate((np.array([0]*len(pop0)), np.array([1]*len(pop1))), axis=0)
                assert len(X) == len(y)
            else:
                popI = get_snps_by_pop_and_set(pops[i], which_set, window_size=window_size)
                X = np.vstack((X, popI))
                y = np.concatenate((y, np.array([i]*len(popI))), axis=0)
                assert len(X) == len(y)
        seed = 42
        np.random.seed(seed)
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        X = X[idxs]
        y = y[idxs]
        log.info(f'Storing {which_set} hdf5...')
        h5f = h5py.File(f'{opath}/{which_set}{int(window_size/1000)}K.h5', 'w')
        h5f.create_dataset('snps', data=X)
        h5f.create_dataset('populations', data=y)
        h5f.close()
    log.info('Done.\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())