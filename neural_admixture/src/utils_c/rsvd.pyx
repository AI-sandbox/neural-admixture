# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, infer_types=True

import time
import logging
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

np.import_array()

# -----------------------------------------------------------------------------
# Cython kernels
# -----------------------------------------------------------------------------

cpdef inline void _multiply_A_omega_uint8_float_parallel(
        const np.uint8_t[:, ::1] A_view,
        const np.float32_t[:, ::1] Omega_view,
        np.float32_t[:, ::1] Y_view,
        int n_rows_A,
        int m_cols_A,
        int k_prime) nogil:
    cdef:
        int i, j, l
        float temp_sum

    for i in prange(n_rows_A, nogil=True, schedule="static", chunksize=16):
        for j in range(k_prime):
            temp_sum = 0.0
            for l in range(m_cols_A):
                temp_sum = temp_sum + <float>A_view[i, l] * Omega_view[l, j]
            Y_view[i, j] = temp_sum

cpdef inline void _multiply_QT_A_float_uint8_parallel(
        const np.float32_t[:, ::1] QT_view,
        const np.uint8_t[:, ::1] A_view,
        np.float32_t[:, ::1] B_view,
        int k_prime_rows_QT,
        int n_rows_A,
        int m_cols_A) nogil:
    cdef:
        int i, j, l
        float temp_sum

    for i in prange(k_prime_rows_QT, nogil=True, schedule='guided'):
        for j in range(m_cols_A):
            temp_sum = 0.0
            for l in range(n_rows_A):
                temp_sum = temp_sum + QT_view[i, l] * <float>A_view[l, j]
            B_view[i, j] = temp_sum

# -----------------------------------------------------------------------------
# Python-callable wrappers
# -----------------------------------------------------------------------------

def multiply_A_omega(np.ndarray[np.uint8_t, ndim=2, mode="c"] A_np,
                     np.ndarray[np.float32_t, ndim=2, mode="c"] Omega_np):
    """
    Multiplica A_np (n_rows_A x m_cols_A) por Omega_np (m_cols_A x k_prime)
    retornando Y_np (n_rows_A x k_prime).
    """
    cdef int n_rows_A = A_np.shape[0]
    cdef int m_cols_A = A_np.shape[1]
    cdef int k_prime = Omega_np.shape[1]
    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] Y_np = \
        np.zeros((n_rows_A, k_prime), dtype=np.float32)

    _multiply_A_omega_uint8_float_parallel(
        A_np, Omega_np, Y_np,
        n_rows_A, m_cols_A, k_prime)
    return Y_np


def multiply_QT_A(np.ndarray[np.float32_t, ndim=2, mode="c"] QT_np,
                  np.ndarray[np.uint8_t, ndim=2, mode="c"] A_np):
    """
    Multiplica QT_np (k_prime x n_rows_A) por A_np (n_rows_A x m_cols_A)
    retornando B_np (k_prime x m_cols_A).
    """
    cdef int k_prime_rows_QT = QT_np.shape[0]
    cdef int n_rows_A = A_np.shape[0]
    cdef int m_cols_A = A_np.shape[1]
    if QT_np.shape[1] != n_rows_A:
        raise ValueError(
            "Dimensiones incompatibles: QT_np.shape[1] debe ser igual a A_np.shape[0]")

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] B_np = \
        np.zeros((k_prime_rows_QT, m_cols_A), dtype=np.float32)

    _multiply_QT_A_float_uint8_parallel(
        QT_np, A_np, B_np,
        k_prime_rows_QT, n_rows_A, m_cols_A)
    return B_np
