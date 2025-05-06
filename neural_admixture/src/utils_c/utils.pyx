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