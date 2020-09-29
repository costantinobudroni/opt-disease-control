import numpy as np
from numba import  jit,  prange, float64, int32,  types


from params_funct import *
from evolution_funct import *
from policy_funct import *
from actions_cont_phys_dist import *
from adam import adam_iteration
    
    


####################################################################################
####################################################################################
#
#                               OPTIMIZATION
#
####################################################################################
####################################################################################

    

    
##simple gradient method to minimize the action, given a lagrangian
##t_out is a tuple indicating the minimum and maximum time since now (t_0) and 
##the outbreak of the disease
def optimize_policy(action, action_deriv, t_0, t_f, confi_vector, x_0, 
                    params = average_values_params(), 
                    lr = 0.01, num_iter=500):
    """initialize vectors for Adam"""
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        """compute action gradient"""
        gradient = action_deriv(t_0, t_f, confi_vector,  x_0, params)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step

                
        if iteration % 200 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))



##Optimization via stochastic gradient descent for noise on the parameters
def optimize_policy_stoch(action, action_deriv_stoch, t_0, t_f, confi_vector, x_0, 
                          params = average_values_params(), uncert = uncertainty_interval(),
                          lr = 0.01, num_iter=500, num_fluct=256, noise_level=0.05):
    """initialize vectors for Adam"""
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        """compute action gradient"""
        gradient = action_deriv_stoch(t_0, t_f, confi_vector,  x_0, params, uncert, num_fluct, noise_level)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step

                
        if iteration % 10000 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))


##Optimization via stochastic gradient descent for noise on the parameters, keeping fixed the first part of the policy
##Necessary for the optimization of policies in presence of uncertainty decreasing with time 
def optimize_policy_stoch_previous_policy(action, action_deriv_stoch_previous, t_0, t_f, previous_confi,  confi_vector,
                                          x_0, params = average_values_params(), uncert = uncertainty_interval(),
                                          lr = 0.01, num_iter=500, num_fluct=256, noise_level=0.05):
    ## initialize vectors for Adam
    Nw = len(confi_vector)
    confi = confi_vector.copy()
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    print("New optimization with noise level {}".format(noise_level))
    for iteration in range(num_iter):
        ## compute action gradient
        gradient = action_deriv_stoch_previous(t_0, t_f, previous_confi,  confi,  x_0, params,
                                               uncert, num_fluct, noise_level)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi -= step
                
        if iteration % 100 == 0:
            loss = action(t_0, t_f, np.concatenate([previous_confi , confi]),  x_0,
                          params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))
    return confi


##Optimization of policies in presence of uncertainty decreasing with time 
def optimize_stoch_decr_uncert(action, action_deriv_stoch, t_0, t_f, confi_vector, 
                                x_0, lr = 0.01, num_iter=500, num_fluct=256, initial_noise_level=0.25):
    ## number of months
    N = int((t_f-t_0)/28)
    confi = confi_vector.copy()
    
    for k in range(N):
        noise_level = initial_noise_level/np.sqrt(k + 1)
        
        confi[4*k:] = optimize_policy_stoch_previous_policy(action, action_deriv_stoch, t_0, t_f, confi[:4*k],
                                                            confi[4*k:], x_0, params = average_values_params(),
                                                            uncert = uncertainty_interval(),lr = lr, num_iter=num_iter,
                                                            num_fluct=num_fluct, noise_level=noise_level)
        
    return confi



##Optimization of lockdown policies (0,1 valued) via gradient descent
def optimize_policy_binary(action, action_deriv, t_0, t_f, confi_vector, x_0, 
                           params = average_values_params(), 
                           lr = 0.01, batch_size=256, num_iter=500):
    """initialize vectors for Adam"""
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        """compute action gradient"""
        gradient = action_deriv(t_0, t_f, confi_vector,  x_0, params, batch_size)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step

                
        if iteration % 20 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))


##Optimization of policies with continuous time parameter, but a fixed number of possible lockdowns
def optimize_policy_ct(action, action_deriv, t_0, t_f,  yepa, x_0,  
                    params = average_values_params(), lr = 0.01, num_iter=500, epsilon = 0.2):
    """initialize vectors for Adam"""
    m = np.zeros(len(yepa))
    v = np.zeros(len(yepa))
    for iteration in range(num_iter):
        """compute action gradient"""
        gradient = action_deriv(t_0, t_f,  x_0, yepa, params, epsilon)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        #step = lr*gradient
        yepa -= step
        
                
        if iteration % 200 == 0:
            tau = (t_f - t_0)*soft_max(yepa)#.cumsum()
            loss = action(t_0, t_f, tau, x_0, params, epsilon)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))
            #print("tau vector", tau)


    
##Optimization of policies including the deliberate infection of a fraction of a population in order to reach herd
##immunity
def optimize_policy_del_inf(action, action_deriv, t_0, t_f, confi_vector, inoc_vector, x_0, 
                    params = average_values_params(), 
                    ethical=False, lr = 0.01, num_iter=500):
    """initialize vectors for Adam"""
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(2*Nw)
    v = np.zeros(2*Nw)
    for iteration in range(num_iter):
        """compute action gradient"""
        gradient = action_deriv(t_0, t_f, confi_vector, inoc_vector, x_0, params, ethical)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step[:Nw]
        inoc_vector -= step[Nw:]
                
        if iteration % 200 == 0:
            loss = action(t_0, t_f, confi_vector, inoc_vector, x_0, params, ethical)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))
