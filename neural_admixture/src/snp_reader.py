import json
import logging
import numpy as np
import pandas as pd
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

class SNPReader:
    def _read_vcf(self, file):
        log.info('Input format is VCF.')
        import allel
        f_tr = allel.read_vcf(file)
        return np.sum(f_tr['calldata/GT'], axis=2).T/2, f_tr['variants/ID']
    
    def _read_hdf5(self, file):
        log.info('Input format is HDF5.')
        import h5py
        f_tr = h5py.File(file, 'r')
        return f_tr['GT'], f_tr['IDs']
    
    def _read_bed(self, file):
        log.info('Input format is BED.')
        from pandas_plink import read_plink
        snps_df, _, G = read_plink('.'.join(file.split('.')[:-1]))
        return (G.T/2).compute(), snps_df['snp'].tolist()
    
    def _read_pgen(self, file):
        log.info('Input format is PGEN.')
        try:
            import pgenlib as pg
        except ImportError as ie:
            log.error('Cannot read PGEN file as pgenlib is not installed.')
            sys.exit(1)
        except Exception as e:
            raise e
        file_prefix = file.split('.pgen')[0]
        pgen, pvar = str.encode(file), f'{file_prefix}.pvar' # Genotype, sample, variant files
        try:
            ids = pd.read_csv(pvar, sep='\t')['ID'].tolist()
        except KeyError as e:
            log.error('Make sure PVAR file contains a header where the variant ids column is named \'ID\'')
            sys.exit(1)
        except Exception as e:
            log.error(e)
            sys.exit(1)
        pgen_reader = pg.PgenReader(pgen)
        calldata = np.ascontiguousarray(np.empty((pgen_reader.get_variant_ct(), 2*pgen_reader.get_raw_sample_ct())).astype(np.int32))
        pgen_reader.read_alleles_range(0, pgen_reader.get_variant_ct(), calldata)
        return (calldata[:,::2]+calldata[:,1::2]).T/2, ids
    
    def _read_npy(self, file):
        log.info('Input format is NPY. Variant IDs will be read from BIM file.')
        try:
            snps_info = pd.read_csv(f'{file.split(".npy")[0]}.bim', sep='\t', usecols=1, header=None)[1].tolist()
        except FileNotFoundError as fnfe:
            log.error(f'Could not read variant IDs as file {file.split(".npy")[0]} does not exist.')
            sys.exit(1)
        except Exception as e:
            log.error(e)
            sys.exit(1)
        npy = np.load(file)
        assert len(npy.shape) in [2, 3]
        if len(npy.shape) == 2:
            return npy/2, None
        return npy.sum(axis=2)/2, None
    
    def _read_data(self, file):
        if file.endswith('.vcf') or file.endswith('.vcf.gz'):
            G, ids = self._read_vcf(file)
        elif file.endswith('.h5') or file.endswith('.hdf5'):
            G, ids = self._read_hdf5(file)
        elif file.endswith('.bed'):
            G, ids = self._read_bed(file)
        elif file.endswith('.pgen'):
            G, ids = self._read_pgen(file)
        elif file.endswith('.npy'):
            G, ids = self._read_npy(file)
        else:
            log.error('Invalid format. Unrecognized file format. Make sure file ends with .vcf | .vcf.gz | .bed | .pgen | .h5 | .hdf5 | .npy')
            sys.exit(1)
        assert int(G.max()) == 1, 'Only biallelic SNPs are supported. Please make sure multiallelic sites have been removed.'
        return G if np.mean(G) < 0.5 else 1-G, ids
    
    def _process_data(self, G, ids, train_config=None, is_inference=False):
        # TODO: unit test for this function
        if is_inference and train_config is None:
            assert G.min() >= 0, 'As config file does not contain training SNPs information, please make sure data does not contain any missing values.'
            log.warn('Config file does not contain training SNPs information. Will assume data matches perfectly.')
            return G, None
        # TODO: there's probably a (faster) npy-friendly way to do this
        if train_config is None:
            na_mask = G < 0
            if len(na_mask) > 0:
                G[na_mask] = np.nan
            log.info('Computing variant-wise mean...')
            snp_means = np.nanmean(G, axis=0)
            if len(na_mask) > 0:
                log.info('Running mean imputation on missing values...')
                idxs = np.where(na_mask)
                idxs = [(idxs[0][i], idxs[1][i]) for i in range(len(idxs[0]))]
                for i, j in idxs:
                    G[i, j] = snp_means[j]
            snp_info = {ids[i]: {'idx': i, 'mean': snp_means[i].item()} for i in range(len(ids))}
            return G, snp_info
        else:
            log.info(f'Matching {"validation" if not is_inference else "inference"} data with training configuration...')
            means = [x['mean'] for x in sorted(list(train_config.values()), key=lambda x: x['idx'])]
            idxs = [(idx, train_config[id]['idx']) for idx, id in enumerate(ids) if id in train_config.keys()]
            mask_in_G, mask_out_G = list(zip(*idxs))
            num_missing = len(training_config)-len(mask_G)     
            if num_missing/len(training_config) > .5:
                log.error('More than 50% of the SNPs contained in the training data could not be found. Exiting...')
                sys.exit(1)
            na_mask = G[:, mask_G] < 0
            means_matrix = np.broadcast_to(means, (G.shape[0], len(train_config)))
            out_G = np.empty(means_matrix.shape)
            out_G[:, mask_out_G] = np.select([na_mask, np.invert(na_mask)], [means_matrix[:, mask_out_G], G[:, mask_G]])
            if num_missing > 0:
                log.warn(f'{num_missing} variants ({round(100*num_missing/out_G.shape[1], 2)}%) used during training are missing from the data. Mean values will be used instead for these SNPs.')
            else:
                log.info('All SNPs have been matched successfully.')
            G = out_G
            return G, None
        
    
    def read_and_process_data(self, file, train_snps_config=None, is_inference=False):
        G, ids = self._read_data(file)
        G, snp_info = self._process_data(G, ids, train_snps_config, is_inference)
        return G, snp_info

