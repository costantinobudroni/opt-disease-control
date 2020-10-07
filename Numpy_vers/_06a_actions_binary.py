import numpy as np

from _01_params_funct import *
from _02abc_evolution_funct import *
from _03_policy_funct import *

    

###############################################################################
###############################################################################
#
#                          BINARY LOCKDOWN ACTIONS
#
###############################################################################
###############################################################################

        
    
### define the num of days N, week, loss and compute the action as loss plus
### total physical distancing (confinement) 
def action_conf_binary(t_0, t_f, confi_vector,  x_0, params):
    N = int(t_f-t_0)
    x = x_0
    loss = 0.0
    
    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)
        x = x + one_step_evol(t, x, confi_vector[week], params)
        loss += mu*relu(x[C_C] - C_bound)
        
    return loss + 7*np.sum(confi_vector)


def sample_confi_grad(t_0, t_f, confi_vector, x_0, params):
    N = int(t_f - t_0)
    Nw = N // 7 + 1 
    grad_confi = np.zeros(Nw)
    prob_deriv = np.zeros(2*Nw)
    loss = np.zeros(2)
    
    for j in range(len(grad_confi)):
        confi_bin = sample_conf_butj(sigmoid(confi_vector),j)
        prob_deriv[j] = -sigmoid(confi_vector[j])*(1-sigmoid(confi_vector[j]))
        prob_deriv[Nw+j] = sigmoid(confi_vector[j])*(1-sigmoid(confi_vector[j]))
        loss[0] = action_conf_binary(t_0,t_f, confi_bin[:Nw], x_0, params)
        loss[1] = action_conf_binary(t_0,t_f, confi_bin[Nw:], x_0, params)
        grad_confi[j] += prob_deriv[j]*loss[0] + prob_deriv[Nw+j]*loss[1]

    return grad_confi

def action_conf_binary_deriv(t_0, t_f, confi_vector,  x_0, params, batch_size):
    ## Sample the action for a given value of the binary confinement policy
    N = int(t_f-t_0)
    Nw = N // 7 + 1 
    grad_confi = np.zeros((batch_size,Nw))
    
    ## compute the gradient w.r.t. confinement variables (with sampling)
    for b in range(batch_size):
        grad_confi[b]=sample_confi_grad(t_0, t_f, confi_vector, x_0, params)
    
    avg_grad_confi = np.sum(grad_confi, axis = 0)/batch_size + 7*sigmoid(confi_vector)*(1-sigmoid(confi_vector)) 
        
    return avg_grad_confi
    

def action_conf_probabilistic(t_0, t_f, confi_vector, x_0, params):
    N = int(t_f-t_0)
    x = x_0
    loss = 0.0
    confi_bin = sample_conf(sigmoid(confi_vector))

    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)

        x = x + one_step_evol(t, x, confi_bin[week], params)
        loss += mu*relu(x[C_C] - C_bound)
        
    return loss + 7*np.sum(confi_bin)


