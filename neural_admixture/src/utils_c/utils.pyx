# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, log1p, sqrt, fmaxf, fminf, sqrtf
from libc.stdlib cimport calloc, free

# Compute reconstruction matrix
cdef inline double _reconstruct(const double* p, const double* q, const size_t K) noexcept nogil:
    cdef:
        size_t k
        double rec = 0.0
    for k in range(K):
        rec += p[k]*q[k]
    return rec

# Read Bed data file:
cpdef void read_bed(const unsigned char[:,::1] bed_source, unsigned char[:,::1] geno_target) noexcept nogil:
    cdef:
        size_t n_snps = geno_target.shape[0]
        size_t n_samples = geno_target.shape[1]
        size_t byte_count = bed_source.shape[1]
        size_t snp_idx, byte_pos, byte_offset, sample_pos
        unsigned char current_byte, geno_value
        unsigned char[4] lookup_table = [2, 9, 1, 0]
    
    with nogil, parallel():
        for snp_idx in prange(n_snps):
            for byte_pos in range(byte_count):
                current_byte = bed_source[snp_idx, byte_pos]
                sample_pos = byte_pos * 4

                if sample_pos < n_samples:
                    geno_target[snp_idx, sample_pos] = lookup_table[current_byte & 3]
                    
                    if sample_pos + 1 < n_samples:
                        geno_target[snp_idx, sample_pos + 1] = lookup_table[(current_byte >> 2) & 3]
                        
                        if sample_pos + 2 < n_samples:
                            geno_target[snp_idx, sample_pos + 2] = lookup_table[(current_byte >> 4) & 3]
                            
                            if sample_pos + 3 < n_samples:
                                geno_target[snp_idx, sample_pos + 3] = lookup_table[(current_byte >> 6) & 3]

# Log-likelihood calculation
cpdef double loglike(const unsigned char[:,::1] G, double[:,::1] P, const double[:,::1] Q) noexcept nogil:
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t K = Q.shape[1]
        size_t i, j
        double logl = 0.0
        double g_d, rec
        double* p
    for j in prange(M):
        p = &P[j,0]
        for i in range(N):
            if G[j,i] != 9:
                rec = _reconstruct(p, &Q[i,0], K)
                g_d = <double>G[j,i]
                logl += g_d*log(rec) + (2.0-g_d)*log1p(-rec)
    return logl