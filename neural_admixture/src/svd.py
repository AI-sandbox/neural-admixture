import logging
import sys
import time
import numpy as np

from .utils_c import rsvd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# High-level randomized SVD function
# -----------------------------------------------------------------------------

def randomized_svd_uint8_input(A_uint8, k, N, M, oversampling=10, power_iterations=4):
    """
    Randomized SVD para matrices uint8 de forma (n_features, m_samples).
    Retorna Vt_k de forma (k, m_samples).
    """
    k_prime = min(N, M, k + oversampling)

    total_start_time = time.time()
    log.info("    1) Generando Ω y Y = A @ Ω...")
    Omega = np.random.randn(M, k_prime).astype(np.float32)
    Y = rsvd.multiply_A_omega(A_uint8, Omega)
    log.info(f"       Y.shape={Y.shape}, time={time.time() - total_start_time:.4f}s")

    if power_iterations > 0:
        iter_start = time.time()
        for _ in range(power_iterations):
            Q_y, _ = np.linalg.qr(Y, mode='reduced')    # (n, k_prime)
            Q_y = np.ascontiguousarray(Q_y.T)
            B_tmp = rsvd.multiply_QT_A(Q_y, A_uint8)      # (k_prime, m)
            B_tmp = np.ascontiguousarray(B_tmp.T)
            Y = rsvd.multiply_A_omega(A_uint8, B_tmp)     # (n, k_prime)
        log.info(f"       Power iterations time={time.time() - iter_start:.4f}s")

    log.info("    2) QR de Y...")
    qr_start = time.time()
    Q, _ = np.linalg.qr(Y, mode='reduced')            # (n, k_prime)
    log.info(f"       Q.shape={Q.shape}, time={time.time() - qr_start:.4f}s")

    log.info("    3) B = Qᵀ @ A...")
    b_start = time.time()
    Q = np.ascontiguousarray(Q.T)
    B = rsvd.multiply_QT_A(Q, A_uint8)                   # (k_prime, m)
    log.info(f"       B.shape={B.shape}, time={time.time() - b_start:.4f}s")

    log.info("    4) SVD de B...")
    svd_start = time.time()
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    log.info(f"       SVD time={time.time() - svd_start:.4f}s")
    
    log.info("")
    log.info(f"    Total time SVD: {time.time() - total_start_time:.4f}s")
    log.info("")
    return Vt[:k, :]
