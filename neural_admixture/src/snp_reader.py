import logging
import numpy as np
import sys

from .utils_c import utils
from math import ceil
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class SNPReader:
    """Wrapper to read genotype data from several formats
    """

    def _read_bed(self, file: str) -> np.ndarray:
        """Reader wrapper for BED files

        Args:
            file (str): path to file.
            master (bool): Wheter or not this process is the master for printing the output.

        Returns:
            np.ndarray: averaged genotype Dask array of shape (n_samples, n_snps)
        """
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
        
        G = np.zeros((N, M), dtype=np.uint8)
        utils.read_bed(B, G)
        del B
        return G
    
    def _read_pgen(self, file: str, master: bool) -> np.ndarray:
        """Reader wrapper for PGEN files"""
        log.info("    Input format is PGEN.")
        try:
            import pgenlib as pg
        except ImportError:
            if master:
                log.error("    Cannot read PGEN file as pgenlib is not installed.")
            sys.exit(1)

        pgen_reader = pg.PgenReader(str.encode(file))
        num_vars = pgen_reader.get_variant_ct()
        num_samples = pgen_reader.get_raw_sample_ct()

        calldata = np.empty((num_vars, 2 * num_samples), dtype=np.uint8)
        pgen_reader.read_alleles_range(0, num_vars, calldata)

        return np.ascontiguousarray((calldata[:, ::2] + calldata[:, 1::2]).T).astype(np.uint8)
    
    def _read_vcf(self, file: str) -> np.ndarray:
        """Reader wrapper for VCF files

        Args:
            file (str): path to file.
            master (bool): Wheter or not this process is the master for printing the output.

        Returns:
            np.ndarray: averaged genotype array of shape (n_samples, n_snps)
        """
        log.info("    Input format is VCF.")
        import allel
        f_tr = allel.read_vcf(file)
        calldata = f_tr["calldata/GT"].astype(np.uint8)
        return np.ascontiguousarray(np.sum(calldata, axis=2, dtype=np.uint8).T)
        
    def read_data(self, file: str) -> np.ndarray:
        """Wrapper of readers

        Args:
            file (str): path to file
        Returns:
            np.ndarray: averaged genotype numpy array of shape (n_samples, n_snps)
        """
        file_extensions = Path(file).suffixes
    
        if '.bed' in file_extensions:
            G = self._read_bed(file)
        elif '.pgen' in file_extensions:
            G = self._read_pgen(file)
        elif '.vcf' in file_extensions:
            G = self._read_vcf(file)
        else:
            log.error("    Invalid format. Unrecognized file format. Make sure file ends with .bed, .pgen or .vcf .")
            sys.exit(1)
        assert int(G.min()) == 0 and int(G.max()) in (2, 3), "Only biallelic SNPs are supported. Please make sure multiallelic sites have been removed."
        return G if G.mean() < 1 else 2 - G
