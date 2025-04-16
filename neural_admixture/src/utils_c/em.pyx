# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport numpy as np
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, sqrt, pow
from libc.stdlib cimport calloc, free

# Update temporary accumulators for P and Q
cdef inline void _update_temp_factors(float* A, float* B, float* t, const float* p, const float* q, const unsigned char g, const float rec, const size_t K) noexcept nogil:
    cdef:
        size_t k
        float g_f = <float>g
        float a = g_f/rec
        float b = (2.0-g_f)/(1.0-rec)
    for k in range(K):
        A[k] += q[k]*a
        B[k] += q[k]*b
        t[k] += p[k]*(a - b) + b

# EM : Update P
cdef inline void _updateEM_P(float* A, float* B, const float* p, float* P_EM, const size_t K) noexcept nogil:
    cdef:
        size_t k
    for k in range(K):
        P_EM[k] = _clip_to_domain((A[k]*p[k])/(p[k]*(A[k] - B[k]) + B[k]))
        A[k] = 0.0
        B[k] = 0.0

# EM : Update Q
cdef inline void _updateEM_Q(float* T, const float* q, float* Q_EM, const float a, const size_t K) noexcept nogil:
    cdef:
        size_t k
        float totalQ = 0.0
    for k in prange(K, nogil=True, schedule='static'):
        Q_EM[k] = _clip_to_domain(q[k] * T[k] * a)
        totalQ += Q_EM[k]
    for k in prange(K, nogil=True, schedule='static'):
        Q_EM[k] /= totalQ
        T[k] = 0.0

# ADAM: Update P
cpdef void adamUpdateP(float[:,::1] P0, const float[:,::1] P1, 
                      float[:,::1] m_P, float[:,::1] v_P, 
                      const float alpha, const float beta1, const float beta2, 
                      const float epsilon, const int t) noexcept nogil:
    cdef:
        size_t i, j, I = P0.shape[0], J = P0.shape[1]
        float delta, m_hat, v_hat, step
        float beta1_t = pow(beta1, t)
        float beta2_t = pow(beta2, t)
        float m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
        float v_scale = 1.0 / (1.0 - beta2_t) if beta2_t != 1.0 else 1.0
    
    for i in prange(I, nogil=True, schedule='static'):
        for j in range(J):
            # Update moments
            delta = P1[i,j] - P0[i,j]
            m_P[i,j] = beta1 * m_P[i,j] + (1.0 - beta1) * delta
            v_P[i,j] = beta2 * v_P[i,j] + (1.0 - beta2) * delta * delta
            
            # Apply updates
            m_hat = m_P[i,j] * m_scale
            v_hat = v_P[i,j] * v_scale
            step = alpha * m_hat / (sqrt(v_hat) + epsilon)
            P0[i,j] = _clip_to_domain(P0[i,j] + step)

# ADAM: Update Q
cpdef void adamUpdateQ(float[:,::1] Q0, const float[:,::1] Q1, 
                      float[:,::1] m_Q, float[:,::1] v_Q, 
                      const float alpha, const float beta1, const float beta2, 
                      const float epsilon, const int t) noexcept nogil:
    cdef:
        size_t i, j, I = Q0.shape[0], J = Q0.shape[1]
        float delta, m_hat, v_hat, step, sumQ = 0.0
        float beta1_t = pow(beta1, t)
        float beta2_t = pow(beta2, t)
        float m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
        float v_scale = 1.0 / (1.0 - beta2_t) if beta2_t != 1.0 else 1.0
    
    for i in prange(I, nogil=True, schedule='static'):
        sumQ = 0.0
        for j in range(J):
            # Update moments
            delta = Q1[i,j] - Q0[i,j]
            m_Q[i,j] = beta1 * m_Q[i,j] + (1.0 - beta1) * delta
            v_Q[i,j] = beta2 * v_Q[i,j] + (1.0 - beta2) * delta * delta
            
            # Apply updates
            m_hat = m_Q[i,j] * m_scale
            v_hat = v_Q[i,j] * v_scale
            step = alpha * m_hat / (sqrt(v_hat) + epsilon)
            Q0[i,j] = _clip_to_domain(Q0[i,j] + step)

            # Normalization
            sumQ += Q0[i,j]
        
        for j in range(J):
            Q0[i,j] /= sumQ

# EM: Apply parameter update for P
cpdef void P_step(const unsigned char[:,::1] G, float[:,::1] P, float[:,::1] P_EM, 
                       const float[:,::1] Q, float[:,::1] Q_T, float[::1] q_bat, 
                       const unsigned int[::1] s) noexcept nogil:
    cdef:
        size_t M = s.shape[0]
        size_t N = G.shape[1]
        size_t K = Q.shape[1]
        size_t i, j, l, x, y
        float rec
        float* p
        float* A
        float* B
        float* t
        float* q_len
        omp.omp_lock_t mutex
    omp.omp_init_lock(&mutex)
    with nogil, parallel():
        A = <float*>calloc(K, sizeof(float))
        B = <float*>calloc(K, sizeof(float))
        t = <float*>calloc(N * K, sizeof(float))
        q_len = <float*>calloc(N, sizeof(float))

        for j in prange(M, schedule='dynamic'):
            l = s[j]
            p = &P[l, 0]
            for i in range(N):
                if G[l, i] != 9:
                    q_len[i] += 1.0
                    rec = _reconstruct(p, &Q[i, 0], K)
                    _update_temp_factors(A, B, &t[i * K], p, &Q[i, 0], G[l, i], rec, K)
            _updateEM_P(A, B, p, &P_EM[l, 0], K)

        omp.omp_set_lock(&mutex)
        for x in range(N):
            q_bat[x] += q_len[x]
            for y in range(K):
                Q_T[x, y] += t[x * K + y]
        omp.omp_unset_lock(&mutex)

        free(A)
        free(B)
        free(t)
        free(q_len)
    omp.omp_destroy_lock(&mutex)

# EM: Apply parameter update for Q
cpdef void Q_step(const float[:,::1] Q, float[:,::1] Q_EM, float[:,::1] T, 
                      float[::1] q_bat) noexcept nogil:
    cdef:
        size_t N = Q.shape[0]
        size_t K = Q.shape[1]
        size_t i, k
        float a
    for i in prange(N, nogil=True, schedule='dynamic'):
        a = 1.0 / (2.0 * q_bat[i])
        _updateEM_Q(&T[i, 0], &Q[i, 0], &Q_EM[i, 0], a, K)
        q_bat[i] = 0.0

# Clip parameters to domain
cdef inline float _clip_to_domain(const float value) noexcept nogil:
    return fmin(fmax(value, 5e-6), 1.0 - 5e-6)

# Compute reconstruction matrix
cdef inline float _reconstruct(const float* p, const float* q, const size_t K) noexcept nogil:
    cdef:
        size_t k
        float rec = 0.0
    for k in range(K):
        rec += p[k]*q[k]
    return rec