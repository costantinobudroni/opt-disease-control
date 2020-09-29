import numpy as np
from numba import vectorize, jit, njit, prange, float64, int32, int64,  boolean, types
import time

from params_funct import *
from evolution_funct_del_inf import *
from policy_funct import *

    
    

####################################################################################
####################################################################################
#
#                              HERD IMMUNITY ACTIONS
#
####################################################################################
####################################################################################


## action to ensure that C_C is below a level and that S gets close to the herd immunity level
## (depends on both t and x)
@jit(['float64(int32, int32, float64[:], float64[:], float64[:], float64[:], boolean)','float64(int32, int32, float64[:], float64[:], float64[:], float64[:], Omitted(True))'])
def action_herd(t_0, t_f, confi_vector, inoc_vector, x_0, params, ethical=False):
    N = int32(t_f-t_0)
    
    x = x_0
    loss = 0.0
    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)
        x = x + one_step_evol_del_inf(t, x, confinement(confi_vector[week]),
                                     infection(inoc_vector[week],ethical), params)


        loss += mu*relu(x[C_C] - C_bound) + abs(x[S] - critical_susc)
        
    return loss

## action derivative
@jit(['float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:], boolean)','float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:], Omitted(True))']) 
def action_herd_deriv(t_0, t_f, confi_vector, inoc_vector, x_0, params, ethical=False):
    N = int32(t_f-t_0)
    Nw = N // 7 + 1
    x = x_0
    y = np.zeros((13, Nw))
    z = np.zeros((13, Nw))
    grad_confi = np.zeros(Nw)
    grad_inoc = np.zeros(Nw)
    
    for k in range(N -1):
        t = t_0 + k
        """calculate week"""
        week = int(k/7.0)
        """evolve compartments and gradients"""
        x_p, y_p, z_p =  one_step_evol_del_inf_with_grad(t, x, y, z, confinement(confi_vector[week]), infection(inoc_vector[week],ethical), week, params)
        x = x + x_p
        y = y + y_p
        z = z + z_p
        """gradient of critical care constraint"""
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C] + np.sign(x[S] - critical_susc)*y[S]
        grad_inoc += mu*heaviside(x[C_C] - C_bound)*z[C_C] + np.sign(x[S] - critical_susc)*z[S]
    return np.concatenate((grad_confi, grad_inoc), 0)




