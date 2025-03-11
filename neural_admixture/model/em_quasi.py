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

def optimize_parameters(G, P, Q, seed, iterations=1000, batches=32, check=5, tole=1e-3):
    M = G.shape[0]
    s = np.arange(M, dtype=np.uint32)
    batch_M = math.ceil(M / batches)
    rng = np.random.default_rng(seed)
    
    #ADAM variables (first and second moments for P and Q)
    m_P = np.zeros_like(P)
    v_P = np.zeros_like(P)
    m_Q = np.zeros_like(Q)
    v_Q = np.zeros_like(Q)
    t = [0]
    
    # Temporal variables
    P1 = np.zeros_like(P)
    Q1 = np.zeros_like(Q)
    Q_tmp = np.zeros_like(Q)
    q_bat = np.zeros(G.shape[1])
    
    # Variables for best parameters:
    P_old, Q_old = P.copy(), Q.copy()
    L_old, L_pre = float('-inf'), float('-inf')
    ts = time.time()
    
    for it in range(iterations):
        rng.shuffle(s)
        for b in range(batches):
            s_bat = s[b * batch_M : min((b + 1) * batch_M, M)]
            
            adamStep(G, P, Q, Q_tmp, P1, Q1, q_bat, s_bat,
                m_P, v_P, m_Q, v_Q, t,
                alpha=0.0025, beta1=0.80, beta2=0.88, epsilon=1e-8)
        
        L_cur = utils.loglike(G, P, Q)
        if L_cur > L_old:
            P_old[:], Q_old[:], L_old = P.copy(), Q.copy(), L_cur
        
        # Verify convergence:
        if (it + 1) % check == 0:
            log.info(f"({it+1})\tLog-like: {L_cur:.1f}\t({time.time()-ts:.1f}s)")
            if L_cur - L_pre < tole:
                log.info("Converged!")
                log.info(f"Final log-likelihood: {L_cur:.1f}")
                break
            L_pre = L_cur
            ts = time.time()
    
    # Return best parameters:
    return P_old, Q_old
