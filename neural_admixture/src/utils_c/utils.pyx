# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, log1p, sqrt, fmaxf, fminf, sqrtf
from libc.stdlib cimport calloc, free

# Compute reconstruction matrix
cdef inline double _reconstruct(const double* p, const double* q, const size_t K) noexcept nogil:
    cdef:
        size_t k
        double rec = 0.0
    for k in range(K):
        rec += q[k]*p[k]
    return rec

# Log-likelihood calculation
cpdef double loglikelihood(const unsigned char[:,::1] G,
                           double[:,::1] P,
                           const double[:,::1] Q,
                           size_t K,
                           double eps=1e-6) noexcept nogil:
    cdef:
        size_t N = G.shape[0]
        size_t M = G.shape[1]
        size_t i, j
        double logl = 0.0
        double g_d, rec
        double* p
    for j in prange(M):
        p = &P[j,0]
        for i in range(N):
            if G[i,j] != 3:
                rec = _reconstruct(p, &Q[i,0], K)
                rec = fmax(eps, fmin(rec, 1.0 - eps))
                
                g_d = <double>G[i,j]
                g_d = fmax(eps, fmin(g_d, 2.0 - eps))
                
                logl += g_d * log(rec) + (2.0 - g_d) * log1p(-rec)
    return logl

# Read Bed data file:
cpdef void read_bed(const unsigned char[:,::1] bed_source, unsigned char[:,::1] geno_target) noexcept nogil:
    cdef:
        size_t n_snps = geno_target.shape[1]
        size_t n_samples = geno_target.shape[0]
        size_t byte_count = bed_source.shape[1]
        size_t snp_idx, byte_pos, sample_pos
        unsigned char current_byte
        unsigned char[4] lookup_table = [2, 3, 1, 0]

    with nogil, parallel():
        for snp_idx in prange(n_snps):
            for byte_pos in range(byte_count):
                current_byte = bed_source[snp_idx, byte_pos]
                sample_pos = byte_pos * 4

                if sample_pos < n_samples:
                    geno_target[sample_pos, snp_idx] = lookup_table[current_byte & 3]

                    if sample_pos + 1 < n_samples:
                        geno_target[sample_pos + 1, snp_idx] = lookup_table[(current_byte >> 2) & 3]

                        if sample_pos + 2 < n_samples:
                            geno_target[sample_pos + 2, snp_idx] = lookup_table[(current_byte >> 4) & 3]

                            if sample_pos + 3 < n_samples:
                                geno_target[sample_pos + 3, snp_idx] = lookup_table[(current_byte >> 6) & 3]
