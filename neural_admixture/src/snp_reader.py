import dask.array as da
import logging
import numpy as np
import sys
from math import ceil
from .utils_c import utils

from pathlib import Path
from typing import Tuple

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class SNPReader:
    """Wrapper to read genotype data from several formats
    """
    def _read_vcf(self, file: str, master: bool) -> np.ndarray:
        """Reader wrapper for VCF files

        Args:
            file (str): path to file.
            master (bool): Wheter or not this process is the master for printing the output.

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        if master:
            log.info("    Input format is VCF.")
        import allel
        f_tr = allel.read_vcf(file)
        calldata = f_tr["calldata/GT"]
        return np.sum(calldata, axis=2).T/2
    
    def _read_bed(self, file: str, master: bool) -> np.ndarray:
        """Reader wrapper for BED files

        Args:
            file (str): path to file.
            master (bool): Wheter or not this process is the master for printing the output.

        Returns:
            da.core.Array: averaged genotype Dask array of shape (n_samples, n_snps)
        """
        if master:
            log.info("    Input format is BED.")

        file_path = Path(file)
        fam_file = file_path.with_suffix(".fam")
        bed_file = file_path.with_suffix(".bed")
        
        with open(fam_file, "r") as fam:
            N = sum(1 for _ in fam)
        N_bytes = ceil(N / 4)
        
        with open(bed_file, "rb") as bed:
            B = np.fromfile(bed, dtype=np.uint8, offset=3)
        
        assert (B.shape[0] % N_bytes) == 0, "bim file doesn't match!"
        M = B.shape[0] // N_bytes
        B.shape = (M, N_bytes)
        
        G = np.zeros((M, N), dtype=np.uint8)
        utils.expandGeno(B, G)
        del B
        has_missing = bool(np.any(G == 9))
        return G, has_missing
    
    def _read_pgen(self, file: str, master: bool) -> np.ndarray:
        """Reader wrapper for PGEN files

        Args:
            file (str): path to file
            master (bool): Wheter or not this process is the master for printing the output.

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        if master:
            log.info("    Input format is PGEN.")
        try:
            import pgenlib as pg
        except ImportError as ie:
            if master:
                log.error("    Cannot read PGEN file as pgenlib is not installed.")
            sys.exit(1)
        except Exception as e:
            raise e
        pgen = str.encode(file) # Genotype, sample, variant files
        pgen_reader = pg.PgenReader(pgen)
        calldata = np.ascontiguousarray(np.empty((pgen_reader.get_variant_ct(), 2*pgen_reader.get_raw_sample_ct())).astype(np.int32))
        pgen_reader.read_alleles_range(0, pgen_reader.get_variant_ct(), calldata)
        return (calldata[:,::2]+calldata[:,1::2]).T/2
    
    def _read_npy(self, file: str, master: bool) -> np.ndarray:
        """Reader wrapper for NPY files

        Args:
            file (str): path to file
            master (bool): Wheter or not this process is the master for printing the output.

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        if master:
            log.info("    Input format is NPY.")
        calldata = np.load(file)
        assert calldata.ndim in [2, 3]
        if calldata.ndim == 2:
            return calldata/2
        return np.nan_to_num(calldata, nan=0.).sum(axis=2)/2
    
    def read_data(self, file: str, master: bool) -> np.ndarray:
        """Wrapper of readers

        Args:
            file (str): path to file
            imputation (str): imputation method. Should be either 'zero' or 'mean'
            master (bool): Wheter or not this process is the master for printing the output.
            
        Returns:
            da.core.Array: averaged genotype Dask array of shape (n_samples, n_snps)
        """
        file_extensions = Path(file).suffixes

        if '.bed' in file_extensions:
            G, has_missing = self._read_bed(file, master)
        else:
            if master:
                log.error("    Invalid format. Unrecognized file format. Make sure file ends with .bed")
            sys.exit(1)
        return (G if G.mean() < 1 else 2 - G), has_missing