import numpy as np
from numba import  jit,  prange, float64, int32,  types

###############################################################################
###############################################################################
#
#            ADAM ITERATION ALGORITHM FOR GRADIENT DESCENT
#
###############################################################################
###############################################################################

### implements one iteration of Adam
  
def adam_iteration(g, m, v, k, l_r):
    ## parameters of Adam algorithm 
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 10**(-8)
    m2 = beta_1*m +(1 - beta_1)*g
    v2 = beta_2*v +(1 - beta_2)*g**2
    m_h = m2/(1 - beta_1**(k + 1))
    v_h = v2/(1 - beta_2**(k + 1))
    return l_r*m_h/(np.sqrt(v_h) + epsilon), m2, v2 
    
    
