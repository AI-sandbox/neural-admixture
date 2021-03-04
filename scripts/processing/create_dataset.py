import gc
import glob
import h5py
import logging
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def splitter(arr, ratios, seed=123):
    np.random.seed(int(seed))
    arr = np.random.permutation(arr)
    ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
    np.split(arr, ind)
    return [x.tolist() for x in np.split(arr, ind)][:len(ratios)]

def get_snps_by_pop(pop, window_size=5000):
    data_path = '/mnt/gpid07/users/margarita.geleta/data/chr22'
    log.info(f'Fetching SNPs for population {pop}')
    for i, snps_arr in enumerate(glob.glob(f'{data_path}/prepared/{pop}_gen_*/{pop}_mat_vcf_2d.npy')):
        aux = np.load(snps_arr, mmap_mode='r')[:,:window_size]
        if i == 0:
            arr = np.empty((0, aux.shape[1]), int)
        arr = np.vstack((arr, aux))
        del aux
        gc.collect()
    log.info('Done.')
    return arr

def holdout_by_pop(snps, populations, *ratios, seed=123, verbose=True):
    _r = len(ratios)
    _sets = [None] * _r
    _pops = [None] * _r
    for id, pop in enumerate(np.unique(populations)):
        pop_idx = np.where(np.asarray(populations) == pop)[0]
        _subsets = splitter(pop_idx, ratios, seed=seed)
        if verbose: log.info(f'Population #{pop} has {len(pop_idx)} samples.\n'+'-' * 50)
        for i in range(_r):
            if verbose: log.info(f'Subsetting {ratios[i] * 100}%, {len(_subsets[i])} samples.')
            if id == 0:
                _sets[i] = snps[_subsets[i],:]
                _pops[i] = populations[_subsets[i]]
                if verbose: log.info(' ' * 5 + f'Stored {_sets[i].shape} matrix in set #{i}.')
            else:
                _subset_i = snps[_subsets[i],:]
                _subpop_i = populations[_subsets[i]]
                _sets[i] = np.vstack((_sets[i],_subset_i))
                _pops[i] = np.append(_pops[i], _subpop_i, axis=0)
                if verbose: log.info(' ' * 5 + f'Stored {_subset_i.shape} matrix in set #{i}.')
                if verbose: log.info(' ' * 5 + f'In total {_sets[i].shape} matrix in set #{i}.')
        if verbose:
            log.info('\n')
    if verbose:
        log.info('Shuffling each set...\n')
    for i in range(_r):
        np.random.seed(int(seed))
        np.random.shuffle(_sets[i])
        np.random.seed(int(seed))
        np.random.shuffle(_pops[i])
    if verbose:
        log.info('-' * 50 + f'\nTotal counts:\n'+'-' * 50)
        for i in range(_r):
            log.info(f'{_sets[i].shape[0]} samples in set #{i}.')
    return _sets, _pops

def main():
    opath = '/home/usuaris/imatge/albert.dominguez/data/chr22'
    pops = ['EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS']
    window_size = 317408
    for i in range(1,len(pops)):
        if i == 1:
            pop0, pop1 = get_snps_by_pop(pops[0], window_size=window_size), get_snps_by_pop(pops[1], window_size=window_size)
            log.info(f'Holdout for population {pops[0]}')
            xi, li = holdout_by_pop(pop0, np.array([0]*pop0.shape[0]), 0.8, 0.1, 0.1, verbose=False)
            log.info(f'Holdout for population {pops[1]}')
            xj, lj = holdout_by_pop(pop1, np.array([1]*pop1.shape[0]), 0.8, 0.1, 0.1, verbose=False)
            TR = np.vstack((xi[0], xj[0]))
            TRL = np.append(li[0], lj[0], axis=0)
            VD = np.vstack((xi[1], xj[1]))
            VDL = np.append(li[1], lj[1], axis=0)
            TS = np.vstack((xi[2], xj[2]))
            TSL = np.append(li[2], lj[2], axis=0)
        else:
            popI = get_snps_by_pop(pops[i], window_size=window_size)
            log.info(f'Holdout for population {pops[i]}')
            xi, li = holdout_by_pop(popI, np.array([i]*popI.shape[0]), 0.8, 0.1, 0.1, verbose=False)
            TR = np.vstack((TR, xi[0]))
            TRL = np.append(TRL, li[0], axis=0)
            VD = np.vstack((VD, xi[1]))
            VDL = np.append(VDL, li[1], axis=0)
            TS = np.vstack((TS, xi[2]))
            TSL = np.append(TSL, li[2], axis=0)
    seed = 123
    np.random.seed(int(seed))
    np.random.shuffle(TR)
    np.random.seed(int(seed))
    np.random.shuffle(TRL)
    np.random.seed(int(seed))
    np.random.shuffle(VD)
    np.random.seed(int(seed))
    np.random.shuffle(VDL)
    np.random.seed(int(seed))
    np.random.shuffle(TS)
    np.random.seed(int(seed))
    np.random.shuffle(TSL)
    log.info('Storing train hdf5...')
    h5f = h5py.File(f'{opath}/prepared/train{int(window_size/1000)}K.h5', 'w')
    h5f.create_dataset('snps', data=TR)
    h5f.create_dataset('populations', data=TRL)
    h5f.close()
    log.info('Storing validation hdf5...')
    h5f = h5py.File(f'{opath}/prepared/valid{int(window_size/1000)}K.h5', 'w')
    h5f.create_dataset('snps', data=VD)
    h5f.create_dataset('populations', data=VDL)
    h5f.close()
    log.info('Storing test hdf5...')
    h5f = h5py.File(f'{opath}/prepared/test{int(window_size/1000)}K.h5', 'w')
    h5f.create_dataset('snps', data=TS)
    h5f.create_dataset('populations', data=TSL)
    h5f.close()
    log.info('Done.\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())