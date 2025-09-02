# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, log1p, sqrtf
from libc.stdlib cimport calloc, free

ctypedef float f32

cdef f32 PRO_MIN = 1e-5
cdef f32 PRO_MAX = 1.0 - (1e-5)
cdef inline f32 _fmaxf(f32 a, f32 b) noexcept nogil: return a if a > b else b
cdef inline f32 _fminf(f32 a, f32 b) noexcept nogil: return a if a < b else b
cdef inline f32 _clamp3(f32 a) noexcept nogil: return _fmaxf(PRO_MIN, _fminf(a, PRO_MAX))

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


cdef inline f32 _computeR(const f32* a, const f32* b, const Py_ssize_t I) noexcept nogil:
    cdef:
        size_t i
        f32 r = 0.0
        f32 c
    for i in range(I):
        c = a[i] - b[i]
        r += c*c
    return r

cpdef f32 rmse(f32[:,::1] A, f32[:,::1] B) noexcept nogil:
    cdef:
        Py_ssize_t N = A.shape[0]
        Py_ssize_t K = A.shape[1]
        f32 r
    r = _computeR(&A[0,0], &B[0,0], N*K)
    return sqrtf(r/(<f32>(N)*<f32>(K)))

cdef inline void _nrmQ(f32* q, const Py_ssize_t K) noexcept nogil:
    cdef:
        size_t k
        f32 sumQ = 0.0
        f32 a, b
    for k in range(K):
        a = q[k]
        b = _clamp3(a)
        sumQ += b
        q[k] = b
    for k in range(K):
        q[k] /= sumQ

cpdef void projectQ(f32[:,::1] Q) noexcept nogil:
    cdef:
        Py_ssize_t N = Q.shape[0]
        Py_ssize_t K = Q.shape[1]
        size_t i
    for i in prange(N, schedule='guided'):
        _nrmQ(&Q[i,0], K)

cpdef void projectP(f32[:,::1] P) noexcept nogil:
    cdef:
        Py_ssize_t M = P.shape[0]
        Py_ssize_t K = P.shape[1]
        size_t j, k
        f32 a
        f32* p
    for j in prange(M, schedule='guided'):
        p = &P[j,0]
        for k in range(K):
            a = p[k]
            p[k] = _clamp3(a)