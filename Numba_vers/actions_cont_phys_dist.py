import numpy as np
from numba import  jit,  prange, float64, int32,  types


from params_funct import *
from evolution_funct import *
from policy_funct import *

    
## Fix the number of compartments
Num_comp = 9


####################################################################################
####################################################################################
#
#                     CONTINUOUS PHYSICAL DISTANCE ACTIONS
#
####################################################################################
####################################################################################


@jit('float64(int32, int32, float64[:], float64[:],  float64[:])', nopython=True, cache=True)
def action_conf(t_0, t_f, confi_vector, x_0, params):
    N = int32(t_f-t_0)
    x = x_0
    loss = 0.0
    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)
        x = x + one_step_evol(t, x, confinement(confi_vector[week]), params)
        loss += mu*relu(x[C_C] - C_bound)
        
    return loss + total_confinement(confi_vector)

###action derivative
@jit('float64[:](int32, int32, float64[:], float64[:],  float64[:])',nopython=True,cache=True)
def action_conf_deriv(t_0, t_f, confi_vector,  x_0, params):
    N = int(t_f-t_0)
    Nw = N // 7 + 1
    x = x_0
    y = np.zeros((Num_comp, Nw))
    grad_confi = np.zeros(Nw)
    grad_inoc = np.zeros(Nw)

    
    for k in range(N -1):
        t = t_0 + k
        """calculate week"""
        week = int(k/7.0)
        """evolve compartments and gradients"""
        x_p, y_p = one_step_evol_with_grad(t, x, y, confinement(confi_vector[week]), week, params)
        x = x + x_p
        y = y + y_p
        """gradient of critical care constraint"""
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C]
    """add gradient related to the explicit presence of confinement"""
    grad_confi += 7*sigmoid(confi_vector)*(1-sigmoid(confi_vector))
    return grad_confi 


@jit('float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:], int32, float64)',nopython=True, cache=True, parallel=True)
def action_conf_deriv_stoch(t_0, t_f, confi_vector,  x_0, params, uncert,
                            num_fluct, noise_level):

    N = int(t_f-t_0)
    Nw = N // 7 + 1

    gradient = np.zeros((num_fluct,Nw))
    st_params= np.zeros((num_fluct,len(params)))
                        
    for k in prange(num_fluct):
        st_params[k] = rand_params_singv(params, uncert, noise_level)
        gradient[k] = action_conf_deriv(t_0, t_f, confi_vector, x_0,
                                        st_params[k])

    avg_grad = np.sum(gradient, axis=0)/num_fluct
    return avg_grad

@jit('float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:],  float64[:], int32, float64)',
     nopython=True, cache=True, parallel=True)
def action_conf_deriv_stoch_previous_policy(t_0, t_f, previous_confi,  confi_vector,  x_0, params, uncert,
                            num_fluct, noise_level):

    
    Nw = len(confi_vector)

    gradient = np.zeros((num_fluct,Nw))
    st_params= np.zeros((num_fluct,len(params)))
    """number of days of previous policy"""
    t_int = t_0 + len(previous_confi)*7.0
                            
    for k in prange(num_fluct):
        st_params[k] = rand_params_singv(params, uncert, noise_level)
        if t_int > t_0 + 1:
            x = future_intervention(x_0, t_0, t_int, previous_confi,  st_params[k])
        else:
            x = x_0
        gradient[k] = action_conf_deriv(t_int, t_f, confi_vector, x,
                                        st_params[k])

    avg_grad = np.sum(gradient, axis=0)/num_fluct
    return avg_grad


