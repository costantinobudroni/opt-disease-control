import numpy as np

from _01_params_funct import *
from _02abc_evolution_funct import *
from _03_policy_funct import *

    
###############################################################################
###############################################################################
#
#                    CONTINUOUS PHYSICAL DISTANCING ACTIONS
#
###############################################################################
###############################################################################

## Fix the number of compartments
Num_comp = 9

### define the num of days N, week, loss and compute the action as loss plus
### total confinement 
def action_conf(t_0, t_f, confi_vector, x_0, params):
    N = int(t_f-t_0)
    x = x_0
    loss = 0.0
    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)
        x = x + one_step_evol(t, x, confinement(confi_vector[week]), params)
        loss += mu*relu(x[C_C] - C_bound)
        
    return loss + total_confinement(confi_vector)



### derivative of the action
def action_conf_deriv(t_0, t_f, confi_vector,  x_0, params):
    N = int(t_f-t_0)
    Nw = N // 7 + 1
    x = x_0
    y = np.zeros((Num_comp, Nw))
    grad_confi = np.zeros(Nw)
    grad_inoc = np.zeros(Nw)
    
    for k in range(N -1):
        t = t_0 + k
        ## the current week 
        week = int(k/7.0)
        ## evolve the compartments and gradients 
        
        x_p, y_p = one_step_evol_with_grad(t, x, y, confinement(confi_vector[week]), week, params)
        x = x + x_p
        y = y + y_p
        
        ## gradient of the critical care constraint 
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C]
        
    ## add graident of the physical distancing measures 
    grad_confi += 7*sigmoid(confi_vector)*(1-sigmoid(confi_vector))
    return grad_confi 




###############################################################################
#                               STOCHASTIC
###############################################################################
    
def action_conf_deriv_stoch(t_0, t_f, confi_vector,  x_0, params, uncert, num_fluct, noise_level):
    
    ## number of days and current week 
    N = int(t_f-t_0)
    Nw = N // 7 + 1

    gradient = np.zeros((num_fluct,Nw))
    st_params= np.zeros((num_fluct,len(params)))
    
    ## create many parameter sets (many = num_fluct) 
    ## compute num_fluct gradients 
    for k in range(num_fluct):
        st_params[k] = rand_params_singv(params, uncert, noise_level)
        gradient[k] = action_conf_deriv(t_0, t_f, confi_vector, x_0,
                                        st_params[k])
    ## average over num_fluct gradients 
    avg_grad = np.sum(gradient, axis=0)/num_fluct
    return avg_grad


###############################################################################
#               STOCHASTIC USING PREVIOUS POLICY CHUNK
###############################################################################
    
def action_conf_deriv_stoch_previous_policy(t_0, t_f, previous_confi,  confi_vector, 
                        x_0, params, uncert, num_fluct, noise_level):

    
    Nw = len(confi_vector)

    gradient = np.zeros((num_fluct,Nw))
    st_params= np.zeros((num_fluct,len(params)))
    ## number of days of previous policy"""
    t_int = t_0 + len(previous_confi)*7.0
                            
    for k in range(num_fluct):
        st_params[k] = rand_params_singv(params, uncert, noise_level)
        if t_int > t_0 + 1:
            x = future_intervention(x_0, t_0, t_int, previous_confi,  st_params[k])
        else:
            x = x_0
        gradient[k] = action_conf_deriv(t_int, t_f, confi_vector, x,
                                        st_params[k])

    avg_grad = np.sum(gradient, axis=0)/num_fluct
    return avg_grad


