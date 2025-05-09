import dask.array as da
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
        utils.read_bed(B, G)
        del B
        has_missing = bool(np.any(G == 3))
        return G, has_missing
    
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