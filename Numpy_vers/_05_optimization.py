import numpy as np

from _01_params_funct import *
from _02abc_evolution_funct import *
from _02d_evolution_funct_del_inf import *
from _03_policy_funct import *
# from actions_cont_phys_dist import *
from _04_adam import adam_iteration
    
    
###############################################################################
###############################################################################
#
#                               OPTIMIZATION
#
###############################################################################
###############################################################################
    



###############################################################################
#                               REGULAR
###############################################################################

    
### optimization via gradient method to minimize an action given a lagrangian
### t_out is a tuple indicating the minimum and maximum time since now (t_0) and 
### the outbreak of the disease

### Nw = number of weeks
def optimize_policy(action, action_deriv, t_0, t_f, confi_vector, x_0, 
                    params = average_values_params(), 
                    lr = 0.01, num_iter=500):
    
    ## initialize vectors for Adam algorithm
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        
        ## compute the gradient of the action
        gradient = action_deriv(t_0, t_f, confi_vector,  x_0, params)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step
        
        ## print the loss function every 200 iterations        
        if iteration % 200 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss)
                                                , num_iter - iteration - 1))


###############################################################################
#                               VACCINATION
###############################################################################
            

### Nw = number of weeks
def optimize_policy_vax(action, action_deriv, t_0, t_f, confi_vector, vax_vector, x_0, 
                    params = average_values_params(), 
                    lr = 0.01, num_iter=500):
    
    ## initialize vectors for Adam algorithm
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(2*Nw)
    v = np.zeros(2*Nw)
    for iteration in range(num_iter):
        
        ## compute the gradient of the action
        gradient = action_deriv(t_0, t_f, confi_vector, vax_vector, x_0, params)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step[:Nw]
        vax_vector -= step[Nw:]

        
        ## print the loss function every 200 iterations        
        if iteration % 200 == 0:
            loss = action(t_0, t_f, confi_vector, vax_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss)
                                                , num_iter - iteration - 1))
            #print("Gradient:", gradient)



            
###############################################################################
#                               STOCHASTIC (PARAMETERS)
###############################################################################

### optimization via stochastic gradient descent for the case that the 
### parameters have noise 
def optimize_policy_stoch(action, action_deriv_stoch, t_0, t_f, confi_vector, x_0, 
                          params = average_values_params(), uncert = uncertainty_interval(),
                          lr = 0.01, num_iter=500, num_fluct=256, noise_level=0.05):
    
    ## initialize vectors for Adam algorithm
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv_stoch(t_0, t_f, confi_vector,  x_0, params, uncert, num_fluct, noise_level)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step
        
        ## print the loss function every 10000 iterations       
        if iteration % 10000 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))


###############################################################################
#                               STOCHASTIC (POLICY)
###############################################################################

### optimization via stochastic gradient descent for the case that the 
### parameters have noise 
def optimize_policy_stoch_pol(action, action_deriv_stoch, t_0, t_f, confi_vector, x_0, 
                          params = average_values_params(), 
                          lr = 0.01, num_iter=500, num_fluct=256, noise_level=0.05):
    
    ## initialize vectors for Adam algorithm
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv_stoch(t_0, t_f, confi_vector,  x_0, params,  num_fluct, noise_level)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step

        ## print the loss function every 1000 iterations       
        if iteration % 10000 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))



###############################################################################
#                               STOCHASTIC (EVOLUTION)
###############################################################################

### optimization via stochastic gradient descent for the case of a stochastic
### evolution
def optimize_policy_stoch_evo(action, action_deriv, t_0, t_f, confi_vector, x_0, 
                          params = average_values_params(), 
                          lr = 0.01, num_iter=500, num_fluct=256, noise_level=0.05):
    
    ## initialize vectors for Adam algorithm
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv(t_0, t_f, confi_vector,  x_0, params,  num_fluct, noise_level)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step

        ## print the loss function every 1000 iterations       
        if iteration % 100 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params, noise_level)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))



###############################################################################
#               STOCHASTIC USING PREVIOUS POLICY CHUNK
###############################################################################

### for the optimization of policies when the uncertainty of the parametsrs 
### is decreasing with time.
### the first part of the policy is kept fixed
### previous_confi = a policy up to a certain month 
def optimize_policy_stoch_previous_policy(action, action_deriv_stoch_previous, t_0, t_f, previous_confi,  confi_vector,
                                          x_0, params = average_values_params(), uncert = uncertainty_interval(),
                                          lr = 0.01, num_iter=500, num_fluct=256, noise_level=0.05):
    ## initialize vectors for Adam algorithm
    Nw = len(confi_vector)
    confi = confi_vector.copy()
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    print("New optimization with noise level {}".format(noise_level))
    for iteration in range(num_iter):
        ## compute the gradient of the action
        
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


###############################################################################
#               OPTIMIZATION OF BINARY (0/1) POLICIES
###############################################################################

##Optimization of lockdown policies (0,1 valued) via gradient descent
def optimize_policy_binary(action, action_deriv, t_0, t_f, confi_vector, x_0, 
                           params = average_values_params(), 
                           lr = 0.01, batch_size=256, num_iter=500):
    ## initialize vectors for Adam algorithm 
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(Nw)
    v = np.zeros(Nw)
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv(t_0, t_f, confi_vector,  x_0, params, batch_size)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step

                
        if iteration % 200 == 0:
            loss = action(t_0, t_f, confi_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))



##Case of vaccinations
def optimize_policy_binary_vax(action, action_deriv, t_0, t_f, confi_vector, vax_vector, x_0,
                           params = average_values_params(), 
                               lr = 0.01, batch_size=256, num_iter=500, few_ld=True):
    ## initialize vectors for Adam algorithm 
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(2*Nw)
    v = np.zeros(2*Nw)
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv(t_0, t_f, confi_vector, vax_vector, x_0, params, batch_size, few_ld=few_ld)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step[:Nw]
        vax_vector -= step[Nw:]

                
        if iteration % 200 == 0:
            loss = action(t_0, t_f, confi_vector, vax_vector, x_0, params)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))
            #print("Gradient:", gradient)
            


            
###############################################################################
#              OPTIMIZATION OF CONTINUOUS-TIME POLICIES
###############################################################################

## Optimization of policies with continuous time parameter, but a fixed number of possible lockdowns
def optimize_policy_ct(action, action_deriv, t_0, t_f,  yepa,  x_0,  
                    params = average_values_params(), lr = 0.01, num_iter=500, epsilon = 0.2):
    ## initialize vectors for Adam algorithm 
    m = np.zeros(len(yepa))
    v = np.zeros(len(yepa))
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv(t_0, t_f,  x_0, yepa, params, epsilon)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        #step = lr*gradient
        yepa -= step
        
                
        if iteration % 200 == 0:
            tau = (t_f - t_0)*soft_max(yepa)#.cumsum()
            loss = action(t_0, t_f, tau, x_0, params, epsilon)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))
            #print("tau vector", tau)



## Case of vaccinations
def optimize_policy_vax_ct(action, action_deriv, t_0, t_f,  yepa, vax_vector, x_0,  
                    params = average_values_params(), lr = 0.01, num_iter=500, epsilon = 0.2):
    ## initialize vectors for Adam algorithm
    Nw = int((t_f -t_0)/7.)+1
    Nc = len(yepa) ## number of changes from no-lockdown to lockdown
    m = np.zeros(Nw+Nc)
    v = np.zeros(Nw+Nc)
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv(t_0, t_f,  x_0, yepa, vax_vector, params, epsilon)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        #step = lr*gradient
        yepa -= step[:Nc]
        vax_vector -= step[Nc:]
        
        
                
        if iteration % 200 == 0:
            tau = (t_f - t_0)*soft_max(yepa)#.cumsum()
            loss = action(t_0, t_f, tau, x_0, vax_vector, params,  epsilon)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))
            #print("tau vector", tau)

            


###############################################################################
#               OPTIMIZATION OF POLICIES WITH DELIBERATE INFECTION
###############################################################################

##Optimization of policies including the deliberate infection of a fraction of a population in order to reach herd
##immunity
def optimize_policy_del_inf(action, action_deriv, t_0, t_f, confi_vector, inoc_vector, x_0, 
                    params = average_values_params(), 
                    ethical=False, lr = 0.01, num_iter=500):
    ## initialize vectors for Adam algorithm 
    Nw = int((t_f -t_0)/7.)+1
    m = np.zeros(2*Nw)
    v = np.zeros(2*Nw)
    for iteration in range(num_iter):
        ## compute gradient of the action 
        gradient = action_deriv(t_0, t_f, confi_vector, inoc_vector, x_0, params, ethical)
        step, m, v = adam_iteration(gradient, m, v, iteration, lr)
        confi_vector -= step[:Nw]
        inoc_vector -= step[Nw:]
                
        if iteration % 200 == 0:
            loss = action(t_0, t_f, confi_vector, inoc_vector, x_0, params, ethical)
            print("Current loss: {}, {} iterations to go.".format(float(loss), num_iter - iteration - 1))
