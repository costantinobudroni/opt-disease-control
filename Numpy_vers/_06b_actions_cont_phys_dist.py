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
#                               VACCINATION
###############################################################################

def action_conf_vax(t_0, t_f, confi_vector, vax_vector, x_0, params):
    N = int(t_f-t_0)
    x = x_0
    loss = 0.0
    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)
        x = x + one_step_evol_vax(t, x, confinement(confi_vector[week]), vaccine(vax_vector[week]), params)
        loss += mu*relu(x[C_C] - C_bound)
        
    return loss + total_confinement(confi_vector) + mu_v*relu(total_vaccines(vax_vector) - V)



### derivative of the action
def action_conf_vax_deriv(t_0, t_f, confi_vector, vax_vector, x_0, params):
    N = int(t_f-t_0)
    Nw = N // 7 + 1
    
    x = x_0
    y = np.zeros((Num_comp, Nw))
    z = np.zeros((Num_comp, Nw))
    grad_confi = np.zeros(Nw)
    grad_vax = np.zeros(Nw)
    
    for k in range(N -1):
        t = t_0 + k
        ## the current week 
        week = int(k/7.0)
        ## evolve the compartments and gradients 
        
        x_p, y_p, z_p = one_step_evol_vax_with_grad(t, x, y, z, confinement(confi_vector[week]), vaccine(vax_vector[week]), week, params)
        x = x + x_p
        y = y + y_p
        z = z + z_p
        
        ## gradient of the critical care constraint
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C]
        grad_vax += mu*heaviside(x[C_C] - C_bound)*z[C_C]
        
    ## add graident of the physical distancing measures 
    grad_confi += 7*sigmoid(confi_vector)*(1-sigmoid(confi_vector))
    ## add gradient related to the vaccine bound
    grad_vax += mu_v*heaviside(total_vaccines(vax_vector) - V)*7*Lambda_v*sigmoid(vax_vector)*(1-sigmoid(vax_vector))
    return np.concatenate((grad_confi, grad_vax),0)




###############################################################################
#                               STOCHASTIC (PARAMETERS)
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
#                               STOCHASTIC (POLICY)
###############################################################################

def action_conf_deriv_stoch_pol(t_0, t_f, confi_vector,  x_0, params,  num_fluct, noise_level):
    
    ## number of days and current week 
    N = int(t_f-t_0)
    Nw = N // 7 + 1

    gradient = np.zeros((num_fluct,Nw))
    st_confi= np.zeros((num_fluct,len(confi_vector)))
    uncert = np.ones(len(confi_vector))/2
    
    ## create many parameter sets (many = num_fluct) 
    ## compute num_fluct gradients 
    for k in range(num_fluct):
        st_confi[k] = rand_params_singv(confi_vector, uncert, noise_level)
        gradient[k] = action_conf_deriv(t_0, t_f, st_confi[k], x_0,
                                        params)
    ## average over num_fluct gradients 
    avg_grad = np.sum(gradient, axis=0)/num_fluct
    return avg_grad



###############################################################################
#                               STOCHASTIC (EVOLUTION)
###############################################################################



### define the num of days N, week, loss and compute the action as loss plus
### total confinement 
def action_conf_stoch_evo(t_0, t_f, confi_vector, x_0, params, noise_level):
    N = int(t_f-t_0)
    x = x_0
    loss = 0.0
    for k in range(N -1):
        t = t_0 + k
        week = int(k/7.)
        ## create noise with with gaussian distribution of mean 0 and std 1
        noise = np.random.normal(0,1)*noise_level
        
        x_p =  one_step_evol(t, x, confinement(confi_vector[week]), params)
        ## Fluctuation proportional to the value of the increment
        ##(e.g., 5% of current value increment for noise_level = 0.05)
        noise = x_p[0]*noise
        
        x = x + x_p
        ## Act only on the first and second compartments
        x[0] = x[0] + noise
        x[1] = x[1] - noise

        loss += mu*relu(x[C_C] - C_bound)
        
    return loss + total_confinement(confi_vector)



### derivative of the action
def action_conf_deriv_stoch_evo(t_0, t_f, confi_vector,  x_0, params, noise_level):
    N = int(t_f-t_0)
    Nw = N // 7 + 1
    x = x_0.copy()
    y = np.zeros((Num_comp, Nw))
    grad_confi = np.zeros(Nw)
    grad_inoc = np.zeros(Nw)
    
    for k in range(N -1):
        t = t_0 + k
        ## the current week 
        week = int(k/7.0)
        ## evolve the compartments and gradients add different noise at each time step

        ## create noise with gaussian distribution of mean 0 and std 1
        noise = np.random.normal(0,1)*noise_level
        
        x_p, y_p = one_step_evol_with_grad(t, x, y, confinement(confi_vector[week]), week, params)
        ## Fluctuation proportional to the value of the increment
        ##(e.g., 5% of current value increment for noise_level = 0.05)

        ## compute noise=beta*I*S*xi
        noise = x_p[0]*noise
        
        x = x + x_p
        ## Act only on the first and second compartments
        x[0] = x[0] + noise
        x[1] = x[1] - noise

        y = y + y_p
        
        ## gradient of the critical care constraint 
        grad_confi += mu*heaviside(x[C_C] - C_bound)*y[C_C]
        
    ## add graident of the physical distancing measures 
    grad_confi += 7*sigmoid(confi_vector)*(1-sigmoid(confi_vector))
    #print(grad_confi)
    return grad_confi 


def action_conf_avg_stoch_evo(t_0, t_f, confi_vector,  x_0, params,  num_fluct, noise_level):
    
    ## number of days and current week 
    N = int(t_f-t_0)
    Nw = N // 7 + 1

    gradient = np.zeros((num_fluct,Nw))

    ## create num_fluct stochastic evolutions
    ## compute num_fluct gradients 
    for k in range(num_fluct):
        gradient[k] = action_conf_deriv_stoch_evo(t_0, t_f, confi_vector, x_0,
                                                  params, noise_level)

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
    ## number of days of previous policy 
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


