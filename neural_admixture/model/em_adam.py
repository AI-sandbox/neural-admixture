import time
import sys
import numpy as np
import logging
import math

from ..src.utils_c import utils, em

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# Función principal estilo ADAM
def adamStep(G, P0, Q0, Q_tmp, P1, Q1, Q_bat, s, 
             m_P, v_P, m_Q, v_Q, t, 
             alpha, beta1, beta2, epsilon):
    # Single EM step
    em.accelBatchP(G, P0, P1, Q0, Q_tmp, Q_bat, s)
    em.accelBatchQ(Q0, Q1, Q_tmp, Q_bat)
    
    # Update ADAM moments
    em.updateAdamMomentsP(P0, P1, m_P, v_P, beta1, beta2)
    em.updateAdamMomentsQ(Q0, Q1, m_Q, v_Q, beta1, beta2)
    
    # Apply ADAM updates
    t_val = t[0] + 1
    em.applyAdamUpdateP(P0, m_P, v_P, alpha, beta1, beta2, epsilon, t_val)
    em.applyAdamUpdateQ(Q0, m_Q, v_Q, alpha, beta1, beta2, epsilon, t_val)
    
    t[0] = t_val

def optimize_parameters(G, P, Q, seed, iterations=1000, batches=32, check=4, tole=1e-3):
    M = G.shape[0]
    s = np.arange(M, dtype=np.uint32)
    batch_M = math.ceil(M / batches)
    rng = np.random.default_rng(seed)
    
    # ADAM variables (first and second moments for P and Q)
    m_P = np.zeros_like(P, dtype=np.float32)
    v_P = np.zeros_like(P, dtype=np.float32)
    m_Q = np.zeros_like(Q, dtype=np.float32)
    v_Q = np.zeros_like(Q, dtype=np.float32)
    t = [0]
    
    # Temporal variables
    P1 = np.zeros_like(P, dtype=np.float32)
    Q1 = np.zeros_like(Q, dtype=np.float32)
    Q_tmp = np.zeros_like(Q, dtype=np.float32)
    q_bat = np.zeros(G.shape[1], dtype=np.float32)
        
    # Variables for best parameters:
    P_old, Q_old = P.copy(), Q.copy()
    
    # Parameters for convergence tracking
    L_old = float('-inf')  # Inicializado con -inf en lugar de calcular
    L_bat = L_pre = float('-inf')
    ts = time.time()
    
    # Early stopping variables
    worsen_count = 0
    L_best_check = float('-inf')
    
    log.info(f"    Using {batches} mini-batches...")
    for it in range(iterations):
        if batches > 1:
            rng.shuffle(s)  # Shuffle SNP order
            for b in range(batches):
                s_bat = s[b * batch_M : min((b + 1) * batch_M, M)]
                
                # Standard updates
                adamStep(G, P, Q, Q_tmp, P1, Q1, q_bat, s_bat,
                        m_P, v_P, m_Q, v_Q, t,
                        alpha=0.0025, beta1=0.80, beta2=0.88, epsilon=1e-8)
        else:
            # Full updates (no mini-batches)
            adamStep(G, P, Q, Q_tmp, P1, Q1, q_bat, s,
                    m_P, v_P, m_Q, v_Q, t,
                    alpha=0.0025, beta1=0.80, beta2=0.88, epsilon=1e-8)
        
        # Convergence or halving check
        if (it + 1) % check == 0:
            if batches > 1:
                L_cur = utils.loglike(G, P.astype(np.float64), Q.astype(np.float64))
                log.info(f"    Iteration {it+1}: \tLog-like: {L_cur:.1f}\t({time.time()-ts:.3f}s)")
                
                # Primera iteración: inicializar valores de referencia
                if L_pre == float('-inf'):
                    L_pre = L_cur
                    L_bat = L_cur
                    L_best_check = L_cur
                    # Actualizar mejores parámetros
                    P_old[:], Q_old[:] = P.copy(), Q.copy()
                    L_old = L_cur
                    ts = time.time()
                    continue
                
                # Early stopping check - if likelihood worsens twice
                if L_cur < L_best_check:
                    worsen_count += 1
                    if worsen_count >= 2:
                        log.info("    Stopping early: Log-likelihood worsened twice consecutively.")
                        # Use best parameters
                        P[:], Q[:] = P_old.copy(), Q_old.copy()
                        L_cur = L_old
                        log.info(f"    Final log-likelihood: {L_old:.1f}")
                        break
                else:
                    worsen_count = 0
                    L_best_check = L_cur
                
                # Check for halving
                if (L_cur < L_bat) or (abs(L_cur - L_bat) < tole):
                    batches = batches // 2  # Halve number of batches
                    if batches > 1:
                        log.info(f"    Using {batches} mini-batches...")
                        L_bat = float('-inf')
                        batch_M = math.ceil(M / batches)
                        L_pre = L_cur
                    else:
                        # Turn off mini-batch acceleration
                        log.info(f"    Using {batches} batch...")
                else:
                    L_bat = L_cur
                    if L_cur > L_old:  # Update best estimates
                        P_old[:], Q_old[:] = P.copy(), Q.copy()
                        L_old = L_cur
            else:
                L_cur = utils.loglike(G, P.astype(np.float64), Q.astype(np.float64))
                log.info(f"    Iteration {it+1}: \tLog-like: {L_cur:.1f}\t({time.time()-ts:.3f}s)")
                
                # Primera iteración: inicializar valores de referencia
                if L_pre == float('-inf'):
                    L_pre = L_cur
                    L_best_check = L_cur
                    # Actualizar mejores parámetros
                    P_old[:], Q_old[:] = P.copy(), Q.copy()
                    L_old = L_cur
                    ts = time.time()
                    continue
                
                # Early stopping check - if likelihood worsens twice
                if L_cur < L_best_check:
                    worsen_count += 1
                    if worsen_count >= 2:
                        log.info("")
                        log.info("    Stopping early: Log-likelihood worsened twice consecutively.")
                        log.info("")
                        # Use best parameters
                        P[:], Q[:] = P_old.copy(), Q_old.copy()
                        L_cur = L_old
                        log.info(f"    Final log-likelihood: {L_old:.1f}")
                        log.info("")
                        break
                else:
                    worsen_count = 0
                    L_best_check = L_cur
                
                # Check for convergence
                if abs(L_cur - L_pre) < tole:
                    if L_cur < L_old:  # Use best estimates
                        P[:], Q[:] = P_old.copy(), Q_old.copy()
                        L_cur = L_old
                    log.info("    Converged!")
                    log.info(f"    Final log-likelihood: {L_cur:.1f}")
                    break
                else:
                    L_pre = L_cur
                    if L_cur > L_old:  # Update best estimates
                        P_old[:], Q_old[:] = P.copy(), Q.copy()
                        L_old = L_cur
            
            ts = time.time()
    
    # Return best parameters (not necessarily the current ones)
    return P_old, Q_old