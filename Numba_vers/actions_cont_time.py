import numpy as np
from numba import vectorize, guvectorize, jit, njit, prange, cuda, float32, float64, int32, int64, uint16, uint32, boolean, types
import numba as nb
import matplotlib.pyplot as plt
import time



from params_funct import *
from evolution_funct import *
from policy_funct import *

## Fix the number of compartments
Num_comp = 9

####################################################################################
####################################################################################
#
#                          CONTINUOUS-TIME LOCKDOWN ACTIONS
#
####################################################################################
####################################################################################


@jit('float64(int32, int32, float64[:], float64[:], float64[:], float64)',  nopython=True,cache=True)
def action_conf_ct(t_0, t_f, taus, x_0, params, epsilon):
    N = int((t_f-t_0)/epsilon) 
    x = x_0.copy() 
    loss = 0.0
    tot_conf = 0.
    for k in range(N -1):

        #if k == 100:
        #    print("action_conf x_0:",x_0)
        t = t_0 + epsilon*k
        conf = confinement_tau(epsilon*k,taus)
        x = x + epsilon*one_step_evol_ct(t, x, conf, params)
        loss += mu*relu(x[C_C] - C_bound)*epsilon
        tot_conf += conf
        
    return loss + tot_conf*epsilon
    

"""action derivative"""
@jit('float64[:](int32, int32, float64[:], float64[:],  float64[:], float64)',  nopython=True,cache=True)
def action_conf_deriv_ct(t_0, t_f,  x_0, yepa, params, epsilon):
    N = int((t_f-t_0)/epsilon)
    
    
    x = x_0.copy() 
    y = np.zeros((Num_comp, len(yepa))) ## we are using only 9 compartments instead of 13
    grad_confi = np.zeros(len(yepa))
    
    taus = (t_f - t_0)*soft_max(yepa)#.cumsum() ##cumulative sum already taken into account, maybe not a good idea?
    last_tau = -1
    for k in range(N -1):
        t = t_0 + k*epsilon

        #if k == 100:
        #    print("action_conf_deriv x_0:",x_0)
        """decide if the gradient should jump"""
        surpass = surpassed_tau(k*epsilon, taus)
        if surpass > last_tau + 1:
            u_min = last_tau + 1
            u_max = surpass 
            
            last_tau = u_max
            y += leap(t, t_0, t_f, x, u_min, u_max, yepa, params)
        
        """evolve compartments and gradients continuously"""
        x_p, y_p = one_step_evol_ct_with_grad(t, x, y, confinement_tau(k*epsilon,taus), params)
        x += x_p*epsilon
        y += y_p*epsilon
        
        
        
        """gradient of critical care constraint"""
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C]*epsilon
    """add gradient related to the explicit presence of confinement"""
    grad_confi += ((t_f-t_0)*soft_max_diff(yepa)[1::2, :]).sum(axis=0)
    return grad_confi



