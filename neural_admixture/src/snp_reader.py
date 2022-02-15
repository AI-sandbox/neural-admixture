import logging
import numpy as np
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

class SNPReader:
    def _read_vcf(self, file):
        log.info('Input format is VCF.')
        import allel
        f_tr = allel.read_vcf(file)
        return np.sum(f_tr['calldata/GT'], axis=2).T/2
    
    def _read_hdf5(self, file):
        log.info('Input format is HDF5.')
        import h5py
        f_tr = h5py.File(file, 'r')
        return f_tr['snps']
    
    def _read_bed(self, file):
        log.info('Input format is BED.')
        from pandas_plink import read_plink
        _, _, G = read_plink('.'.join(file.split('.')[:-1]))
        return (G.T/2).compute()
    
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
        pgen, _, _ = str.encode(file), f'{file_prefix}.psam', f'{file_prefix}.pvar' # Genotype, sample, variant files
        pgen_reader = pg.PgenReader(pgen)
        calldata = np.ascontiguousarray(np.empty((pgen_reader.get_variant_ct(), 2*pgen_reader.get_raw_sample_ct())).astype(np.int32))
        pgen_reader.read_alleles_range(0, pgen_reader.get_variant_ct(), calldata)
        return (calldata[:,::2]+calldata[:,1::2]).T/2
    
    def _read_npy(self, file):
        log.info('Input format is NPY.')
        npy = np.load(file)
        assert len(npy.shape) in [2, 3]
        if len(npy.shape) == 2:
            return npy/2
        return npy.sum(axis=2)/2
    
    def read_data(self, file):
        if file.endswith('.vcf') or file.endswith('.vcf.gz'):
            G = self._read_vcf(file)
        elif file.endswith('.h5') or file.endswith('.hdf5'):
            G = self._read_hdf5(file)
        elif file.endswith('.bed'):
            G = self._read_bed(file)
        elif file.endswith('.pgen'):
            G = self._read_pgen(file)
        elif file.endswith('.npy'):
            G = self._read_npy(file)
        else:
            log.error('Invalid format. Unrecognized file format. Make sure file ends with .vcf | .vcf.gz | .bed | .pgen | .h5 | .hdf5 | .npy')
            sys.exit(1)
        assert int(G.min()) == 0 and int(G.max()) == 1, 'Only biallelic SNPs are supported. Please make sure multiallelic sites have been removed.'
        return G if np.mean(G) < 0.5 else 1-G