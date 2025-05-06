import dask.array as da
import logging
import numpy as np
import sys

from math import ceil

from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class SNPReader:
    """Wrapper to read genotype data from several formats
    """

    def _read_bed(self, file: str, master: bool):
        file_path = Path(file)
        fam_file = file_path.with_suffix('.fam')
        bed_file = file_path.with_suffix('.bed')
        
        with open(fam_file, "r") as f:
            N = sum(1 for _ in f)
        N_bytes = ceil(N/4)
        B = np.fromfile(bed_file, dtype=np.uint8, offset=3)
        M = B.size // N_bytes
        B.shape = (M, N_bytes)

        # desempacar 2 bits → 4 genotipos
        shifts = np.array([0,2,4,6], dtype=np.uint8)
        codes = (B[..., None] >> shifts) & 3
        lookup = np.array([2,9,1,0], dtype=np.uint8)
        G = lookup[codes].reshape(M, N_bytes*4)[:,:N]
        has_missing = bool(np.any(G==9))
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