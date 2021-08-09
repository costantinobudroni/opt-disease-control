import numpy as np

from numba import jit,  prange, float64, int32,  types

from _01_params_funct import *
from _02abc_evolution_funct import *
from _03_policy_funct import *


###############################################################################
###############################################################################
#
#                          CONTINUOUS-TIME LOCKDOWN ACTIONS
#
###############################################################################
###############################################################################

## Fix the number of compartments
Num_comp = 9

###############################################################################
#                          WITHOUT VACCINATION
###############################################################################



### define the num of days N, week, loss and compute the action as loss plus
### total confinement 

@jit('float64(int32, int32, float64[:], float64[:], float64[:], float64)',  nopython=True,cache=True)
def action_conf_ct(t_0, t_f, taus, x_0, params, epsilon):
    N = int((t_f-t_0)/epsilon) 
    x = x_0.copy() 
    loss = 0.0
    tot_conf = 0.
    for k in range(N -1):

        t = t_0 + epsilon*k
        conf = confinement_tau(epsilon*k,taus)
        x = x + epsilon*one_step_evol_ct(t, x, conf, params)
        loss += mu*relu(x[C_C] - C_bound)*epsilon
        tot_conf += conf
        
    return loss + tot_conf*epsilon
    

### derivative of the action
@jit('float64[:](int32, int32, float64[:], float64[:],  float64[:], float64)',  nopython=True,cache=True)
def action_conf_deriv_ct(t_0, t_f,  x_0, yepa, params, epsilon):
    N = int((t_f-t_0)/epsilon)

    x = x_0.copy() 
    y = np.zeros((Num_comp, len(yepa))) 
    grad_confi = np.zeros(len(yepa))
    
    taus = (t_f - t_0)*soft_max(yepa)
    last_tau = -1
    for k in range(N -1):
        t = t_0 + k*epsilon

        ## decide if the gradient should jump 
        surpass = surpassed_tau(k*epsilon, taus)
        if surpass > last_tau + 1:
            u_min = last_tau + 1
            u_max = surpass - 1
            
            last_tau = u_max
            y += leap(t, t_0, t_f, x, u_min, u_max, yepa, params)
        
        ## evolve the compartments and gradients continuously 
        x_p, y_p = one_step_evol_ct_with_grad(t, x, y, confinement_tau(k*epsilon,taus), params)
        x += x_p*epsilon
        y += y_p*epsilon
 
        ## gradient of the critical care constraint 
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C]*epsilon
   ## add gradient related to the physical distancing 
    grad_confi += ((t_f-t_0)*soft_max_diff(yepa)[1::2, :]).sum(axis=0)
    return grad_confi



###############################################################################
#                          WITH VACCINATION
###############################################################################



@jit('float64(int32, int32, float64[:], float64[:], float64[:], float64[:], float64)',  nopython=True,cache=True)
def action_conf_vax_ct(t_0, t_f, taus, x_0, vax_vector, params, epsilon):
    N = int((t_f-t_0)/epsilon) 
    x = x_0.copy() 
    loss = 0.0
    tot_conf = 0.
    for k in range(N -1):

        t = t_0 + epsilon*k
        week = int((t-t_0)//7)
        conf = confinement_tau(epsilon*k,taus)
        x = x + epsilon*one_step_evol_vax_ct(t, x, conf, vaccine(vax_vector[week]), params)
        loss += mu*relu(x[C_C] - C_bound)*epsilon
        tot_conf += conf
        
    return loss + tot_conf*epsilon
    

### derivative of the action
@jit('float64[:](int32, int32, float64[:], float64[:], float64[:],  float64[:], float64)',  nopython=True,cache=True)
def action_conf_vax_deriv_ct(t_0, t_f,  x_0, yepa, vax_vector, params, epsilon):
    N = int((t_f-t_0)/epsilon)
    Nw = int((t_f-t_0) // 7) + 1
    
    x = x_0.copy() 
    y = np.zeros((Num_comp, len(yepa)))
    z = np.zeros((Num_comp, Nw))
                 
    grad_confi = np.zeros(len(yepa))
    grad_vax = np.zeros(Nw)
    
    taus = (t_f - t_0)*soft_max(yepa)
    last_tau = -1
    for k in range(N -1):
        t = t_0 + k*epsilon

        week = int((t-t_0)//7)
        ## decide if the gradient should jump 
        surpass = surpassed_tau(k*epsilon, taus)
        if surpass > last_tau + 1:
            u_min = last_tau + 1
            u_max = surpass - 1
            
            last_tau = u_max
            y += leap(t, t_0, t_f, x, u_min, u_max, yepa, params)
        
        ## evolve the compartments and gradients continuously 
        x_p, y_p, z_p= one_step_evol_vax_ct_with_grad(t, x, y, z, confinement_tau(k*epsilon,taus), vaccine(vax_vector[week]), week, params)
        x += x_p*epsilon
        y += y_p*epsilon
        z += z_p*epsilon
                 
        ## gradient of the critical care constraint 
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C]*epsilon
        grad_vax += mu*heaviside(x[C_C] - C_bound)*z[C_C]*epsilon
   ## add gradient related to the physical distancing 
    grad_confi += ((t_f-t_0)*soft_max_diff(yepa)[1::2, :]).sum(axis=0)
    grad_vax += mu_v*heaviside(total_vaccines(vax_vector) - V)*7*Lambda_v*sigmoid(vax_vector)*(1-sigmoid(vax_vector))

    return np.concatenate((grad_confi,grad_vax), 0)



