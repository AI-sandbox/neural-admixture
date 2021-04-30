import gc
import glob
import h5py
import logging
import numpy as np
import pickle
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

data_path = '/mnt/gpid08/users/albert.dominguez/data/chr22'
# with open(f'{data_path}/unlinked_indices_2gen_0.05.pkl', 'rb') as fb:
#         unlinked_idxs = pickle.load(fb)

def get_snps_by_pop_and_set(pop, which_set):
    log.info(f'Fetching SNPs for population {pop} and set {which_set}') 
    for i, snps_arr in enumerate([f'{data_path}/{which_set}/{pop}/gen_0/mat_vcf_2d.npy',f'{data_path}/{which_set}/{pop}/gen_2/mat_vcf_2d.npy']):
        aux = np.load(snps_arr, mmap_mode='r')#[:,unlinked_idxs][:,::2]
        if i == 0:
            arr = np.empty((0, aux.shape[1]), np.int8)
        assert aux.shape[0] % 2 == 0, 'Number of samples in NPY should be even'
        arr = np.vstack((arr, aux[0::2]+aux[1::2]))
        #arr = np.vstack((arr, aux))
        del aux
        gc.collect()
    log.info('Done.')
    return arr

def main():
    opath = '/mnt/gpid08/users/albert.dominguez/data/chr22/windowed'
    pops = ['AFR', 'AMR', 'EAS', 'EUR', 'OCE', 'SAS', 'WAS']
    for which_set in ['train', 'valid']:
        for i in range(1,len(pops)):
            if i == 1:
                pop0, pop1 = get_snps_by_pop_and_set(pops[0], which_set), get_snps_by_pop_and_set(pops[1], which_set)
                X = np.vstack((pop0, pop1))
                y = np.concatenate((np.array([0]*len(pop0)), np.array([1]*len(pop1))), axis=0)
                assert len(X) == len(y)
            else:
                popI = get_snps_by_pop_and_set(pops[i], which_set)
                X = np.vstack((X, popI))
                y = np.concatenate((y, np.array([i]*len(popI))), axis=0)
                assert len(X) == len(y)
        log.info(f'Storing {which_set} hdf5...')
        h5f = h5py.File(f'{opath}/{which_set}_2gen_sum.h5', 'w')
        h5f.create_dataset('snps', data=X)
        h5f.create_dataset('populations', data=y)
        h5f.close()
    log.info('Done.\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())
