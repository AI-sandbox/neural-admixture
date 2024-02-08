import dask.array as da
import logging
import numpy as np
import sys

from pathlib import Path
from typing import Literal

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

class SNPReader:
    """Wrapper to read genotype data from several formats
    """
    def _read_vcf(self, file: str) -> np.ndarray:
        """Reader wrapper for VCF files

        Args:
            file (str): path to file.

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        log.info('Input format is VCF.')
        import allel
        f_tr = allel.read_vcf(file)
        calldata = f_tr["calldata/GT"]
        return np.sum(calldata, axis=2).T/2
    
    def _read_hdf5(self, file: str) -> np.ndarray:
        """Reader wrapper for HDF5 files. HDF5 should directly contain the averaged genotype array

        Args:
            file (str): path to file.

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        log.info('Input format is HDF5.')
        import h5py
        f_tr = h5py.File(file, 'r')
        return f_tr['snps']
    
    def _read_bed(self, file: str) -> da.core.Array:
        """Reader wrapper for BED files

        Args:
            file (str): path to file.

        Returns:
            da.core.Array: averaged genotype Dask array of shape (n_samples, n_snps)
        """
        log.info('Input format is BED.')
        from pandas_plink import read_plink
        _, _, G = read_plink(str(Path(file).with_suffix("")))
        return (G.T/2)
    
    def _read_pgen(self, file: str) -> np.ndarray:
        """Reader wrapper for PGEN files

        Args:
            file (str): path to file

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        log.info('Input format is PGEN.')
        try:
            import pgenlib as pg
        except ImportError as ie:
            log.error('Cannot read PGEN file as pgenlib is not installed.')
            sys.exit(1)
        except Exception as e:
            raise e
        pgen = str.encode(file) # Genotype, sample, variant files
        pgen_reader = pg.PgenReader(pgen)
        calldata = np.ascontiguousarray(np.empty((pgen_reader.get_variant_ct(), 2*pgen_reader.get_raw_sample_ct())).astype(np.int32))
        pgen_reader.read_alleles_range(0, pgen_reader.get_variant_ct(), calldata)
        return (calldata[:,::2]+calldata[:,1::2]).T/2
    
    def _read_npy(self, file: str) -> np.ndarray:
        """Reader wrapper for NPY files

        Args:
            file (str): path to file

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        log.info('Input format is NPY.')
        calldata = np.load(file)
        assert calldata.ndim in [2, 3]
        if calldata.ndim == 2:
            return calldata/2
        return np.nan_to_num(calldata, nan=0.).sum(axis=2)/2
    
    def read_data(self, file: str, imputation: str) -> da.core.Array:
        """Wrapper of readers

        Args:
            file (str): path to file
            imputation (str): imputation method. Should be either 'zero' or 'mean'

        Returns:
            da.core.Array: averaged genotype Dask array of shape (n_samples, n_snps)
        """
        file_extensions = Path(file).suffixes
        if '.vcf' in file_extensions:
            G = self._read_vcf(file)
        elif '.h5' in file_extensions or '.hdf5' in file_extensions:
            G = self._read_hdf5(file)
        elif '.bed' in file_extensions:
            G = self._read_bed(file)
        elif '.pgen' in file_extensions:
            G = self._read_pgen(file)
        elif '.npy' in file_extensions:
            G = self._read_npy(file)
        else:
            log.error('Invalid format. Unrecognized file format. Make sure file ends with .vcf | .vcf.gz | .bed | .pgen | .h5 | .hdf5 | .npy')
            sys.exit(1)
        if isinstance(G, np.ndarray):
            G = da.from_array(G)
        G = self._impute(G, method=imputation)
        assert int(G.min().compute()) == 0 and int(G.max().compute()) == 1, 'Only biallelic SNPs are supported. Please make sure multiallelic sites have been removed.'
        return G if G.mean().compute() < 0.5 else 1-G

    @staticmethod
    def _impute(G: da.core.Array, method: Literal["zero", "mean"]="mean") -> da.core.Array:
        """Impute missing values

        Args:
            G (da.core.Array): genotype array

        Returns:
            da.core.Array: imputed genotype array
        """
        if da.isnan(G).any().compute():
            log.warning(f"Data contains missing values. Will perform {method}-imputation.")
            if method == "zero":
                return da.nan_to_num(G, 0.)
            elif method == "mean":
                snp_means = da.nanmean(G, axis=0)[None, :]
                return da.where(da.isnan(G), snp_means, G)
            else:
                raise ValueError("Invalid imputation method. Only 'zero' and 'mean' are supported.")
        return G
