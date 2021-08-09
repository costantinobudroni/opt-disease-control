import numpy as np
from numba import  jit,  prange, float64, int32,  boolean, types

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

        






###############################################################################
#                         NO VACCINATION
###############################################################################


### define the num of days N, week, loss and compute the action as loss plus
### total physical distancing (confinement) 
@jit('float64(int32, int32, float64[:], float64[:],  float64[:])', nopython=True,cache=True)
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

@jit('float64[:](int32, int32, float64[:], float64[:], float64[:])',nopython=True,cache=True)
def sample_confi_grad(t_0, t_f, confi_vector, x_0, params):
    N = int(t_f - t_0)
    Nw = N // 7 + 1 
    grad_confi = np.zeros(Nw)
    prob_deriv = np.zeros(2*Nw)
    confi_v = confi_vector.copy()
    loss = np.zeros(2)
    
    for j in range(Nw):
        confi_bin = sample_conf_butj(sigmoid(confi_v),j)
        prob_deriv[j] = -sigmoid(confi_v[j])*(1-sigmoid(confi_v[j]))
        prob_deriv[Nw+j] = sigmoid(confi_v[j])*(1-sigmoid(confi_v[j]))
        loss[0] = action_conf_binary(t_0,t_f, confi_bin[:Nw], x_0, params)
        loss[1] = action_conf_binary(t_0,t_f, confi_bin[Nw:], x_0, params)
        grad_confi[j] += prob_deriv[j]*loss[0] + prob_deriv[Nw+j]*loss[1]

    return grad_confi


@jit('float64[:](int32, int32, float64[:],  float64[:], float64[:], int32)',nopython=True, parallel=True, cache=True)
def action_conf_binary_deriv(t_0, t_f, confi_vector, x_0, params, batch_size):
    ## Sample the action for a given value of the binary confinement policy
    N = int(t_f-t_0)
    Nw = N // 7 + 1 
    grad_confi = np.zeros((batch_size,Nw))
    

    ## compute the gradient w.r.t. confinement variables (with sampling)
    for b in prange(batch_size):
        grad_confi[b]=sample_confi_grad(t_0, t_f, confi_vector,  x_0, params)
    
    avg_grad_confi = np.sum(grad_confi, axis = 0)/batch_size + 7*sigmoid(confi_vector)*(1-sigmoid(confi_vector))

        
    return avg_grad_confi


###############################################################################
#                         VACCINATION
###############################################################################



@jit(['float64(int32, int32, float64[:], float64[:], float64[:],  float64[:], boolean)','float64(int32, int32, float64[:], float64[:], float64[:],  float64[:], Omitted(True))'], nopython=True,cache=True)
def action_conf_binary_vax(t_0, t_f, confi_vector, vax_vector, x_0, params, few_ld=True):
    N = int(t_f-t_0)
    x = x_0.copy()
    Nw = N // 7 + 1
    loss = 0.0
    
    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)
        x = x + one_step_evol_vax(t, x, confi_vector[week], vaccine(vax_vector[week]), params)
        loss += mu*relu(x[C_C] - C_bound)

    if few_ld:
    ## FIX PARAMETERS SUCH AS NUMBER OF CHANGES AND PENALTY FUNCTION 
         num_changes = 10
         penalty_transitions=10**2
         count_changes = np.sum(np.abs(confi_vector[:Nw-1]-confi_vector[1:Nw]))
         loss += penalty_transitions*relu(count_changes - num_changes)

        
    return loss + 7*np.sum(confi_vector)


@jit(['float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:], boolean)', 'float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:], Omitted(True))'],nopython=True,cache=True)
def sample_confi_grad_vax(t_0, t_f, confi_vector, vax_vector, x_0, params, few_ld=True):
    N = int(t_f - t_0)
    Nw = N // 7 + 1
         
    grad_confi = np.zeros(Nw)
    prob_deriv = np.zeros(2*Nw)
    confi_v = confi_vector.copy()
    loss = np.zeros(2)
    
    for j in range(len(grad_confi)):
        confi_bin = sample_conf_butj(sigmoid(confi_v),j)
        prob_deriv[j] = -sigmoid(confi_v[j])*(1-sigmoid(confi_v[j]))
        prob_deriv[Nw+j] = sigmoid(confi_v[j])*(1-sigmoid(confi_v[j]))
        loss[0] = action_conf_binary_vax(t_0,t_f, confi_bin[:Nw], vax_vector, x_0, params) 
        loss[1] = action_conf_binary_vax(t_0,t_f, confi_bin[Nw:], vax_vector, x_0, params)
        grad_confi[j] += prob_deriv[j]*loss[0] + prob_deriv[Nw+j]*loss[1]


    return grad_confi


@jit('float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:])',nopython=True,cache=True)
def sample_vax_grad(t_0, t_f, confi_vector, vax_vector, x_0, params):

    N = int(t_f - t_0)
    Nw = N // 7 + 1 ## we are assuming a weekly policy
    x_vax = x_0.copy() 
    z_vax = np.zeros((Num_comp,Nw))
    grad_vax = np.zeros(Nw)
    confi_v = confi_vector.copy()
    confi_bin = sample_conf(sigmoid(confi_v))
    for k in range(N):
        t = t_0 + k
        week = int(k/7)
        x_p, z_p = one_step_evol_vax_with_grad_binary(t, x_vax, z_vax, confi_bin[week], vaccine(vax_vector[week]), week, params)
        x_vax += x_p
        z_vax += z_p
        grad_vax += mu*heaviside(x_vax[C_C] - C_bound)*z_vax[C_C]


    grad_vax += mu_v*heaviside( total_vaccines(vax_vector) - V)*7*Lambda_v*sigmoid(vax_vector)*(1-sigmoid(vax_vector))

    return grad_vax


@jit(['float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:], int32, boolean)','float64[:](int32, int32, float64[:], float64[:], float64[:], float64[:], int32, Omitted(True))'],nopython=True, parallel=True, cache=True)
def action_conf_binary_vax_deriv(t_0, t_f, confi_vector, vax_vector, x_0, params, batch_size, few_ld=True):
    ## Sample the action for a given value of the binary confinement policy
    N = int(t_f-t_0)
    Nw = N // 7 + 1 ## we are assuming a weekly policy
    x = x_0
    grad_confi = np.zeros((batch_size,Nw))
    grad_vax = np.zeros((batch_size,Nw))
    
    ## compute the gradient w.r.t. confinment variables (with sampling)
    for b in range(batch_size):

        grad_confi[b]=sample_confi_grad_vax(t_0, t_f, confi_vector, vax_vector, x, params,few_ld=few_ld)
    
    avg_grad_confi = np.sum(grad_confi, axis = 0)/batch_size + 7*sigmoid(confi_vector)*(1-sigmoid(confi_vector))
    
    ## compute the gradient w.r.t. vaccination variables (with sampling)
    for b in prange(batch_size):
        grad_vax[b] = sample_vax_grad(t_0, t_f, confi_vector, vax_vector, x, params)
        
    avg_grad_vax = np.sum(grad_vax, axis = 0)/batch_size

    return np.concatenate((avg_grad_confi, avg_grad_vax), 0)


@jit('float64(int32, int32, float64[:], float64[:], float64[:])', nopython=True,cache=True)
def action_conf_probabilistic(t_0, t_f, confi_vector, x_0, params):
    N = int32(t_f-t_0)
    x = x_0
    loss = 0.0
    confi_bin = sample_conf(sigmoid(confi_vector))

    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)

        x = x + one_step_evol(t, x, confi_bin[week], params)
        loss += mu*relu(x[C_C] - C_bound)
        
    return loss + 7*np.sum(confi_bin)


