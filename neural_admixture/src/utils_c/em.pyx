# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, sqrt, pow
from libc.stdlib cimport calloc, free

# Truncate parameters to domain
cdef inline float _project(const float s) noexcept nogil:
    return fmin(fmax(s, 1e-5), 1.0 - 1e-5)

# Estimate individual allele frequencies
cdef inline float _computeH(const float* p, const float* q, const size_t K) noexcept nogil:
    cdef:
        size_t k
        float h = 0.0
    for k in range(K):
        h += p[k]*q[k]
    return h

# Inner loop updates for temp P and Q
cdef inline void _inner(const float* p, const float* q, float* p_a, float* p_b, float* q_thr, const unsigned char g, const float h, const size_t K) noexcept nogil:
    cdef:
        size_t k
        float d = <float>g
        float a = d/h
        float b = (2.0-d)/(1.0-h)
    for k in range(K):
        p_a[k] += q[k]*a
        p_b[k] += q[k]*b
        q_thr[k] += p[k]*(a - b) + b

# Outer loop accelerated update for P
cdef inline void _outerAccelP(const float* p, float* p_n, float* p_a, float* p_b, const size_t K) noexcept nogil:
    cdef:
        size_t k
    for k in range(K):
        p_n[k] = _project((p_a[k]*p[k])/(p[k]*(p_a[k] - p_b[k]) + p_b[k]))
        p_a[k] = 0.0
        p_b[k] = 0.0

# Outer loop accelerated update for Q
cdef inline void _outerAccelQ(const float* q, float* q_new, float* q_tmp, const float a, const size_t K) noexcept nogil:
    cdef:
        size_t k
        float sumQ = 0.0
    for k in prange(K, nogil=True, schedule='static'):
        q_new[k] = _project(q[k] * q_tmp[k] * a)
        sumQ += q_new[k]
    for k in prange(K, nogil=True, schedule='static'):
        q_new[k] /= sumQ
        q_tmp[k] = 0.0

# ADAM: Update moments for P
cpdef void updateAdamMomentsP(float[:,::1] P0, const float[:,::1] P1, float[:,::1] m_P, float[:,::1] v_P, const float beta1, const float beta2) noexcept nogil:
    cdef:
        size_t i, j, I = P0.shape[0], J = P0.shape[1]
        float delta
    for i in prange(I, nogil=True, schedule='guided'):
        for j in range(J):
            delta = P1[i,j] - P0[i,j]
            m_P[i,j] = beta1 * m_P[i,j] + (1.0 - beta1) * delta
            v_P[i,j] = beta2 * v_P[i,j] + (1.0 - beta2) * delta * delta

# ADAM: Update moments for Q
cpdef void updateAdamMomentsQ(float[:,::1] Q0, const float[:,::1] Q1, float[:,::1] m_Q, float[:,::1] v_Q, const float beta1, const float beta2) noexcept nogil:
    cdef:
        size_t i, j, I = Q0.shape[0], J = Q0.shape[1]
        float delta
    for i in prange(I, nogil=True, schedule='guided'):
        for j in range(J):
            delta = Q1[i,j] - Q0[i,j]
            m_Q[i,j] = beta1 * m_Q[i,j] + (1.0 - beta1) * delta
            v_Q[i,j] = beta2 * v_Q[i,j] + (1.0 - beta2) * delta * delta

# ADAM: Apply parameter update for P
cpdef void applyAdamUpdateP(float[:,::1] P0, const float[:,::1] m_P, const float[:,::1] v_P, 
                          const float alpha, const float beta1, const float beta2, 
                          const float epsilon, const int t) noexcept nogil:
    cdef:
        size_t i, j, I = P0.shape[0], J = P0.shape[1]
        float m_hat, v_hat, step
        float beta1_t = pow(beta1, t)
        float beta2_t = pow(beta2, t)
        float m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
        float v_scale = 1.0 / (1.0 - beta2_t) if beta2_t != 1.0 else 1.0
    for i in prange(I, nogil=True, schedule='guided'):
        for j in range(J):
            m_hat = m_P[i,j] * m_scale
            v_hat = v_P[i,j] * v_scale
            step = alpha * m_hat / (sqrt(v_hat) + epsilon)
            P0[i,j] = _project(P0[i,j] + step)

# ADAM: Apply parameter update for Q with normalization
cpdef void applyAdamUpdateQ(float[:,::1] Q0, const float[:,::1] m_Q, 
                           const float[:,::1] v_Q, float alpha, 
                           float beta1, float beta2, float epsilon, 
                           int t) noexcept nogil:
    cdef:
        size_t i, j, I = Q0.shape[0], J = Q0.shape[1]
        float m_hat, v_hat, step, sumQ = 0.0
        float beta1_t = pow(beta1, t)
        float beta2_t = pow(beta2, t)
        float m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
        float v_scale = 1.0 / (1.0 - beta2_t) if beta2_t != 1.0 else 1.0
    for i in prange(I, nogil=True, schedule='guided'):
        sumQ = 0.0
        for j in range(J):
            m_hat = m_Q[i, j] * m_scale
            v_hat = v_Q[i, j] * v_scale
            step = alpha * m_hat / (sqrt(v_hat) + epsilon)
            Q0[i, j] = _project(Q0[i, j] + step)
            sumQ += Q0[i, j]
        for j in range(J):
            Q0[i, j] /= sumQ

### Batch functions
# Update P in batch acceleration
cpdef void accelBatchP(const unsigned char[:,::1] G, float[:,::1] P, float[:,::1] P_new, 
                      const float[:,::1] Q, float[:,::1] Q_tmp, float[::1] q_bat, 
                      const unsigned int[::1] s) noexcept nogil:
    cdef:
        size_t M = s.shape[0]
        size_t N = G.shape[1]
        size_t K = Q.shape[1]
        size_t i, j, l, x, y
        float h
        float* p
        float* p_thr
        float* q_thr
        float* q_len
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        p_thr = <float*>calloc(2*K, sizeof(float))
        q_thr = <float*>calloc(N*K, sizeof(float))
        q_len = <float*>calloc(N, sizeof(float))
        for j in prange(M, schedule='dynamic'):
            l = s[j]
            p = &P[l,0]
            for i in range(N):
                if G[l,i] != 9:
                    q_len[i] += 1.0
                    h = _computeH(p, &Q[i,0], K)
                    _inner(p, &Q[i,0], &p_thr[0], &p_thr[K], &q_thr[i*K], G[l,i], h, K)
            _outerAccelP(p, &P_new[l,0], &p_thr[0], &p_thr[K], K)
        
        # omp critical
        omp.omp_set_lock(&mutex)
        for x in range(N):
            q_bat[x] += q_len[x]
            for y in range(K):
                Q_tmp[x,y] += q_thr[x*K + y]
        omp.omp_unset_lock(&mutex)
        free(p_thr)
        free(q_thr)
        free(q_len)
    omp.omp_destroy_lock(&mutex)

# Batch update Q from temp arrays
cpdef void accelBatchQ(const float[:,::1] Q, float[:,::1] Q_new, float[:,::1] Q_tmp, 
                      float[::1] q_bat) noexcept nogil:
    cdef:
        size_t N = Q.shape[0]
        size_t K = Q.shape[1]
        size_t i, k
        float a
    for i in prange(N, nogil=True, schedule='dynamic'):
        a = 1.0 / (2.0 * q_bat[i])
        _outerAccelQ(&Q[i, 0], &Q_new[i, 0], &Q_tmp[i, 0], a, K)
        q_bat[i] = 0.0