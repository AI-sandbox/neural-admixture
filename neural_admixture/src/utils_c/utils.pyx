# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, log1p, sqrt, fmaxf, fminf, sqrtf
from libc.stdlib cimport calloc, free

# Estimate individual allele frequencies
cdef inline double _computeH(const double* p, const double* q, const size_t K) noexcept nogil:
    cdef:
        size_t k
        double h = 0.0
    for k in range(K):
        h += p[k]*q[k]
    return h

# Expand data from 2-bit to 8-bit genotype matrix
cpdef void expandGeno(const unsigned char[:,::1] B, unsigned char[:,::1] G) noexcept nogil:
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t N_b = B.shape[1]
        size_t i, j, b, x, bit
        unsigned char[4] recode = [2, 9, 1, 0]
        unsigned char mask = 3
        unsigned char byte
    with nogil, parallel():
        for j in prange(M):
            i = 0
            for b in range(N_b):
                byte = B[j,b]
                for bit in range(4):
                    G[j,i] = recode[(byte >> 2*bit) & mask]
                    i = i + 1
                    if i == N:
                        break

cpdef void estimateMean(const unsigned char[:,::1] G, float[::1] mean) noexcept nogil:
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t i, j
        float c, n

    for j in prange(M):  # Iterate through each SNP (row)
        c = 0.0
        n = 0.0
        for i in range(N):  # Iterate through each individual (column)
            if G[j, i] != 9:
                c = c + <float>G[j, i]
                n = n + 1.0
        if n > 0:
            mean[j] = c / n
        else:
            mean[j] = 0.0

# Log-likelihood
cpdef double loglike(const unsigned char[:,::1] G, double[:,::1] P, const double[:,::1] Q) noexcept nogil:
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t K = Q.shape[1]
        size_t i, j
        double res = 0.0
        double d, h
        double* p
    for j in prange(M):
        p = &P[j,0]
        for i in range(N):
            if G[j,i] != 9:
                h = _computeH(p, &Q[i,0], K)
                d = <double>G[j,i]
                res += d*log(h) + (2.0-d)*log1p(-h)
    return res