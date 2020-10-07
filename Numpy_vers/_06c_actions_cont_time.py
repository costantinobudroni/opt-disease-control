import numpy as np
import matplotlib.pyplot as plt
import time

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

### define the num of days N, week, loss and compute the action as loss plus
### total confinement 


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
    ## add gradient related to the explicit presence of confinement
    grad_confi += ((t_f-t_0)*soft_max_diff(yepa)[1::2, :]).sum(axis=0)
    return grad_confi



