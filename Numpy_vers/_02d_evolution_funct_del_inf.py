import numpy as np

from _01_params_funct import *

###############################################################################
###############################################################################
#
#             EVOLUTION FUNCTIONS FOR CASE d : DELIBERATE INFECTION
#
###############################################################################
###############################################################################


### note that 'confinement' should be interpreted as physical distancing


## Fix the number of compartments
Num_comp = 13


### codes to evolve the equation and the gradients
### x is the vector of compartments; y, the derivative of x with respect to confinement; 
### z, the derivative of x with respect to deliberate infection
### week is the current week
def one_step_evol_del_inf_with_grad(t, x, y, z, confi, inoc, week, params):
    ## we assume that confi = sigmoid(s), incor = Lambda_i*sigmoid(mu) 
    rel_R = (params[r]-1)*confi + 1
    ## derivative of relative reproduction number 
    rel_Rp = (params[r]-1)*confi*(1-confi)
    ## derivative of deliberate infection
    inocp = inoc*(1- (inoc/Lambda_i))
    
    xp = np.zeros(x.shape)
    yp = np.zeros(y.shape)
    zp = np.zeros(z.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - inoc*x[S]
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E] + inoc*x[S] 
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*(x[I_H] ) - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*(x[I_C] ) - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*(x[I_R] + x[I_Rp] ) + params[delta_h]*x[H_H] + params[xi_c]*x[C_C]                           
    xp[Ep] = inoc*x[S] - params[nu]*x[Ep]
    xp[I_Rp] = params[nu]*params[p_r]*x[Ep] - params[gamma]*x[I_Rp]
    xp[I_Hp] = params[nu]*params[p_h]*x[Ep] - params[gamma]*x[I_Hp]
    xp[I_Cp] = params[nu]*params[p_c]*x[Ep] - params[gamma]*x[I_Cp]

    
    ## the part that does not depend on confi explicitly

    yp[S] = -rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - inoc*y[S]
    yp[E] = rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*y[E] +inoc*y[S] 
    yp[I_R] = params[p_r]*params[nu]*y[E] - params[gamma]*y[I_R]
    yp[I_H] = params[p_h]*params[nu]*y[E] - params[gamma]*y[I_H]
    yp[I_C] = params[p_c]*params[nu]*y[E] - params[gamma]*y[I_C]
    yp[H_H] = params[gamma]*(y[I_H] ) - params[delta_h]*y[H_H]
    yp[H_C] = params[gamma]*(y[I_C] ) - params[delta_c]*y[H_C]
    yp[C_C] = params[delta_c]*y[H_C] - params[xi_c]*y[C_C]

    yp[R] = params[gamma]*(y[I_R] + y[I_Rp]) + params[delta_h]*y[H_H] + params[xi_c]*y[C_C]

    yp[Ep] = inoc*y[S] - params[nu]*y[Ep]
    yp[I_Rp] = params[nu]*params[p_r]*y[Ep] - params[gamma]*y[I_Rp]
    yp[I_Hp] = params[nu]*params[p_h]*y[Ep] - params[gamma]*y[I_Hp]
    yp[I_Cp] = params[nu]*params[p_c]*y[Ep] - params[gamma]*y[I_Cp]
    yp[S, week] += -beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*rel_Rp
    yp[E, week] += beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*rel_Rp                           

    ## the part that does not depend on inoc explicitly

    zp[S] = -rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - inoc*z[S]
    zp[E] = rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*z[E] +inoc*z[S]
    zp[I_R] = params[p_r]*params[nu]*z[E] - params[gamma]*z[I_R]
    zp[I_H] = params[p_h]*params[nu]*z[E] - params[gamma]*z[I_H]
    zp[I_C] = params[p_c]*params[nu]*z[E] - params[gamma]*z[I_C]
    zp[H_H] = params[gamma]*(z[I_H] ) - params[delta_h]*z[H_H]
    zp[H_C] = params[gamma]*(z[I_C] ) - params[delta_c]*z[H_C]
    zp[C_C] = params[delta_c]*z[H_C] - params[xi_c]*z[C_C]

    zp[R] = params[gamma]*(z[I_R] + z[I_Rp]) + params[delta_h]*z[H_H] \
        + params[xi_c]*z[C_C]
    zp[Ep] = inoc*z[S] - params[nu]*z[Ep]
    zp[I_Rp] = params[nu]*params[p_r]*z[Ep] - params[gamma]*z[I_Rp]
    zp[I_Hp] = params[nu]*params[p_h]*z[Ep] - params[gamma]*z[I_Hp]
    zp[I_Cp] = params[nu]*params[p_c]*z[Ep] - params[gamma]*z[I_Cp]

    ##now, the part with the partial derivative of G^i
    zp[S, week] += - inocp*x[S]
    zp[Ep, week] += inocp*x[S]                           

    return xp, yp, zp

### function which only evolves the vector of compartments
def one_step_evol_del_inf(t, x, confi, inoc, params):
    ## we assume that confi = sigmoid(s), inoc = Lambda_i*sigmoid(mu)
    rel_R = (params[r]-1)*confi + 1
    xp = np.zeros(x.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - inoc*x[S]
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E] +inoc*x[S] 
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*(x[I_H] ) - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*(x[I_C] ) - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*(x[I_R] + x[I_Rp]) + params[delta_h]*x[H_H] + params[xi_c]*x[C_C]
    xp[Ep] = inoc*x[S] - params[nu]*x[Ep]
    xp[I_Rp] = params[nu]*params[p_r]*x[Ep] - params[gamma]*x[I_Rp]
    xp[I_Hp] = params[nu]*params[p_h]*x[Ep] - params[gamma]*x[I_Hp]
    xp[I_Cp] = params[nu]*params[p_c]*x[Ep] - params[gamma]*x[I_Cp]
                           
    
    return xp


### function to evolve the system without intervention, starting from 
### some initial conditions t_0, and ending at t_f
### N is discrete num of days   
### returns the final occupation of the vector of compartments
def future_nat(x_0, t_0, t_f, params):
    N = int(t_f-t_0)
    x = x_0
    #at time t_0, x= x_0!!!
    for k in range(N):
        t = t_0 + k        
        x += one_step_evol_del_inf(t, x, 0.0, 0.0, params)
    return x


### returns the occupation of the compartments at time t_0, assuming that 
### the outbreak happened at time t_0-offset
def initial_pop(offset, t_0, params):
    # compartment occupation at the outbreak: just ten exposed people
    x_0 = np.zeros(Num_comp)
    x_0[S] = 1.0 - 10.0/population
    x_0[E] = 10.0/population
    x = future_nat(x_0, t_0 - offset, t_0, params)
    return x

### evolve the compartment vector x for N days
### create a history of the evolution at each time step in a matrix.
def evolve(x_0, t_0, t_f, params, confinement, infection):
    N = int(t_f-t_0)
    x = np.zeros(list(x_0.shape) + [N])
    x[..., 0] = x_0

    #at time t_0, x= x_0!!!
    for k in range(N - 1):
        t = t_0 + k
        x[:, k + 1] = x[:, k] + one_step_evol_del_inf(t, x[:, k], confinement(t), infection(t), params)
    return x


### natural evolution of the compartments from x_0, t_0 or N days 
### without any intervention whatsoever
def evolve_nat(x_0, t_0, t_f, params):
    def confinement(t):
        return 0.0
    def infection(t):
        return 0.0
    return evolve(x_0, t_0, t_f, params, confinement, infection)
    

### evolves the compartments from x_0, t_0 for N days with a pre-loaded policy
### of physical distancing and infection 
def evolve_with_intervention(poli, x_0, t_0, t_f, params):
    return evolve(x_0, t_0, t_f, params, poli.confinement, poli.infection)


