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

def svd_flip(V, U):
    """
    Adjust signs of V rows based on dominant signs in U columns to ensure consistent SVD output.

    Parameters:
    -----------
    V : np.ndarray
        Matrix (e.g., V or Vt) to flip signs on rows.
    U : np.ndarray
        Left singular vectors matrix to determine sign direction.

    Returns:
    --------
    np.ndarray
        Sign-corrected version of V.
    """
    k_components = U.shape[1]
    max_abs_val_row_indices = np.argmax(np.abs(U), axis=0)
    col_selector = np.arange(k_components)
    elements_for_sign = U[max_abs_val_row_indices, col_selector]
    signs = np.sign(elements_for_sign)
    return V * signs[:, np.newaxis]

def RSVD(A_uint8, N, M, k=8, seed=42, oversampling=10, power_iterations=2):
    """
    Randomized SVD para matrices uint8 de forma (n_features, m_samples).
    Retorna Vt_k de forma (k, m_samples).
    """
    rng = np.random.default_rng(seed)
    k_prime = max(k + oversampling, 20)

    total_start_time = time.time()
    log.info("    1) Generating Ω y Y = A @ Ω...")
    Omega = rng.standard_normal(size=(M, k_prime), dtype=np.float32)
    Y = rsvd.multiply_A_omega(A_uint8, Omega)
    log.info(f"       Time={time.time() - total_start_time:.4f}s")

    if power_iterations > 0:
        iter_start = time.time()
        for _ in range(power_iterations):
            Q_y, _ = np.linalg.qr(Y, mode='reduced')    # (n, k_prime)
            Q_y = np.ascontiguousarray(Q_y.T)
            B_tmp = rsvd.multiply_QT_A(Q_y, A_uint8)      # (k_prime, m)
            B_tmp = np.ascontiguousarray(B_tmp.T)
            Y = rsvd.multiply_A_omega(A_uint8, B_tmp)     # (n, k_prime)
        log.info(f"       Power iterations time={time.time() - iter_start:.4f}s")

    log.info("    2) QR of Y...")
    qr_start = time.time()
    Q, _ = np.linalg.qr(Y, mode='reduced')            # (n, k_prime)
    log.info(f"       Time={time.time() - qr_start:.4f}s")

    log.info("    3) B = Qᵀ @ A...")
    b_start = time.time()
    Q = np.ascontiguousarray(Q.T)
    B = rsvd.multiply_QT_A(Q, A_uint8)                   # (k_prime, m)
    log.info(f"       Time={time.time() - b_start:.4f}s")

    log.info("    4) SVD of B...")
    svd_start = time.time()
    Ut, St, Vt = np.linalg.svd(B, full_matrices=False)
    log.info(f"       SVD time={time.time() - svd_start:.4f}s")
    
    Vt = svd_flip(Vt, Ut)
    
    log.info("")
    log.info(f"    Total time SVD: {time.time() - total_start_time:.4f}s")
    log.info("")
    return Vt[:k, :]