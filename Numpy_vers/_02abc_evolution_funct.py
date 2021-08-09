import numpy as np

from _01_params_funct import *

###############################################################################
###############################################################################
#
#                 EVOLUTION FUNCTIONS FOR CASES abc : 
#          binary; continuous physical distancing; continuous time
#
###############################################################################
###############################################################################

### note that 'confinement' should be interpreted as physical distancing


## Fix the number of compartments
Num_comp = 9


### codes to evolve the dynamical equations and the gradients
### x is the vector of compartments; y, the derivatives with respect to confinement; 
### week is the current week

### Function which evolves the vector of compartments and compute the gradient for confinement policies
def one_step_evol_with_grad(t, x, y,  confi,  week, params):
    ## rel_R is the relative reproduction rate 
    ## confi is the physical distancing parameter, here confi = sigmoid(s)
    rel_R = (params[r]-1)*confi + 1
    ## derivative of relative reproduction rate
    rel_Rp = (params[r]-1)*confi*(1-confi)
    
    xp = np.zeros(x.shape)
    yp = np.zeros(y.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) 
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E] 
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*(x[I_H] ) - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*(x[I_C] ) - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*(x[I_R] ) + params[delta_h]*x[H_H] \
        + params[xi_c]*x[C_C]
    
    ## the part that does not depend explicitly on confi 

    yp[S] = -rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C]))
    yp[E] = rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*y[E] 
    yp[I_R] = params[p_r]*params[nu]*y[E] - params[gamma]*y[I_R]
    yp[I_H] = params[p_h]*params[nu]*y[E] - params[gamma]*y[I_H]
    yp[I_C] = params[p_c]*params[nu]*y[E] - params[gamma]*y[I_C]
    yp[H_H] = params[gamma]*(y[I_H] ) - params[delta_h]*y[H_H]
    yp[H_C] = params[gamma]*(y[I_C] ) - params[delta_c]*y[H_C]
    yp[C_C] = params[delta_c]*y[H_C] - params[xi_c]*y[C_C]

    yp[R] = params[gamma]*(y[I_R] ) + params[delta_h]*y[H_H] + params[xi_c]*y[C_C]

    yp[S, week] += -beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*rel_Rp
    yp[E, week] += beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*rel_Rp

    
    return xp, yp

### Function which only evolves the vector of compartments for confinement policies
def one_step_evol(t, x, confi, params):
    ## we assume that confi = sigmoid(s)
    rel_R = (params[r]-1)*confi + 1
    xp = np.zeros(x.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) 
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E] 
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*(x[I_H] ) - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*(x[I_C] ) - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*(x[I_R] ) + params[delta_h]*x[H_H] + params[xi_c]*x[C_C]
    
    return xp



#########################
## CASE WITH VACCINATION
#########################

### function which only evolves the vector of compartments for confinement and vaccination policies
def one_step_evol_vax(t, x, confi, vacc, params):
    ## we assume that confi = sigmoid(s), vacc = Lambda_v*sigmoid(v)
    rel_R = (params[r]-1)*confi + 1
    bt = beta(t, params)
    xp = np.zeros(x.shape)

    xp[S] = -rel_R*bt*x[S]*(x[I_R] + x[I_H] +x[I_C]) - vacc*x[S]
    xp[E] = rel_R*bt*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E] 
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*(x[I_H] ) - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*(x[I_C] ) - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*(x[I_R] ) + params[delta_h]*x[H_H] + params[xi_c]*x[C_C] + vacc*x[S]

    if np.isnan(np.sum(x)):
        print("x is NaN")
        print("x:", x)
        print("confi", confi)
        print("vacc", vacc)

    
    return xp


## Case with vaccination continuous physical distance
def one_step_evol_vax_with_grad(t, x, y, z,  confi,  vacc, week, params):
    ## rel_R is the relative reproduction rate 
    ## confi is the physical distancing parameter, here confi = sigmoid(s) and vacc = Lambda_v*sigmoid(v)
    rel_R = (params[r]-1)*confi + 1
    ## derivative of relative reproduction rate
    rel_Rp = (params[r]-1)*confi*(1-confi)
    vaccp = vacc*(1-vacc/Lambda_v)
    
    xp = np.zeros(x.shape)
    yp = np.zeros(y.shape)
    zp = np.zeros(z.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - vacc*x[S]
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E] 
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*(x[I_H] ) - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*(x[I_C] ) - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*(x[I_R] ) + params[delta_h]*x[H_H] + params[xi_c]*x[C_C] + vacc*x[S]
    
    ###first the part that does not depend explicitly on confi 
    yp[S] = -rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - vacc*y[S]
    yp[E] = rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*y[E] 
    yp[I_R] = params[p_r]*params[nu]*y[E] - params[gamma]*y[I_R]
    yp[I_H] = params[p_h]*params[nu]*y[E] - params[gamma]*y[I_H]
    yp[I_C] = params[p_c]*params[nu]*y[E] - params[gamma]*y[I_C]
    yp[H_H] = params[gamma]*(y[I_H] ) - params[delta_h]*y[H_H]
    yp[H_C] = params[gamma]*(y[I_C] ) - params[delta_c]*y[H_C]
    yp[C_C] = params[delta_c]*y[H_C] - params[xi_c]*y[C_C]

    yp[R] = params[gamma]*(y[I_R] ) + params[delta_h]*y[H_H] + params[xi_c]*y[C_C] + vacc*y[S]
    ###now, the part with the partial derivative of G^i
    yp[S, week] += -beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*rel_Rp
    yp[E, week] += beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*rel_Rp

    ###first, the part that does not depend on vacc explicitly
    zp[S] = -rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - vacc*z[S]
    zp[E] = rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*z[E]
    zp[I_R] = params[p_r]*params[nu]*z[E] - params[gamma]*z[I_R]
    zp[I_H] = params[p_h]*params[nu]*z[E] - params[gamma]*z[I_H]
    zp[I_C] = params[p_c]*params[nu]*z[E] - params[gamma]*z[I_C]
    zp[H_H] = params[gamma]*z[I_H] - params[delta_h]*z[H_H]
    zp[H_C] = params[gamma]*z[I_C] - params[delta_c]*z[H_C]
    zp[C_C] = params[delta_c]*z[H_C] - params[xi_c]*z[C_C]
    zp[R] = params[gamma]*z[I_R] + params[delta_h]*z[H_H] + params[xi_c]*z[C_C] + vacc*z[S]
    ###now, the part with the partial derivative of G^i
    zp[S, week] += - vaccp*x[S]
    zp[R,week] += vaccp*x[S]


    
    return xp, yp, zp

## Case with vaccination continuous physical distance
def one_step_evol_vax_with_grad_binary(t, x, z,  confi,  vacc, week, params):
    ## rel_R is the relative reproduction rate 
    ## confi is the physical distancing parameter, here confi = sigmoid(s) and vacc = Lambda_v*sigmoid(v)
    rel_R = (params[r]-1)*confi + 1
    ## derivative of relative reproduction rate
    rel_Rp = (params[r]-1)*confi*(1-confi)
    vaccp = vacc*(1-vacc/Lambda_v)
    
    xp = np.zeros(x.shape)
    zp = np.zeros(z.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - vacc*x[S]
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E] 
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*(x[I_H] ) - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*(x[I_C] ) - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*(x[I_R] ) + params[delta_h]*x[H_H] + params[xi_c]*x[C_C] + vacc*x[S]


    ###first, the part that does not depend on vacc explicitly
    zp[S] = -rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - vacc*z[S]
    zp[E] = rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*z[E]
    zp[I_R] = params[p_r]*params[nu]*z[E] - params[gamma]*z[I_R]
    zp[I_H] = params[p_h]*params[nu]*z[E] - params[gamma]*z[I_H]
    zp[I_C] = params[p_c]*params[nu]*z[E] - params[gamma]*z[I_C]
    zp[H_H] = params[gamma]*z[I_H] - params[delta_h]*z[H_H]
    zp[H_C] = params[gamma]*z[I_C] - params[delta_c]*z[H_C]
    zp[C_C] = params[delta_c]*z[H_C] - params[xi_c]*z[C_C]
    zp[R] = params[gamma]*z[I_R] + params[delta_h]*z[H_H] + params[xi_c]*z[C_C] + vacc*z[S]
    ###now, the part with the partial derivative of G^i
    zp[S, week] += - vaccp*x[S]
    zp[R,week] += vaccp*x[S]


    
    return xp,  zp


##################################################
#### CONTINUOUS-TIME PHYSICAL DISTANCING FUNCTIONS
##################################################

## same as above but for continuous time 

def one_step_evol_ct(t, x, confi, params):    
    rel_R = (params[r]-1)*confi + 1
    xp = np.zeros(x.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E]
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*x[I_H] - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*x[I_C] - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*x[I_R] + params[delta_h]*x[H_H] \
        + params[xi_c]*x[C_C]

    return xp

def one_step_evol_ct_with_grad(t, x, y, confi, params):    
    rel_R = (params[r]-1)*confi + 1
    
    xp = np.zeros(x.shape)
    yp = np.zeros(y.shape)
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E]
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*x[I_H] - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*x[I_C] - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*x[I_R] + params[delta_h]*x[H_H] \
        + params[xi_c]*x[C_C]

    yp[S] = -rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C]))
    yp[E] = rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*y[E]
    yp[I_R] = params[p_r]*params[nu]*y[E] - params[gamma]*y[I_R]
    yp[I_H] = params[p_h]*params[nu]*y[E] - params[gamma]*y[I_H]
    yp[I_C] = params[p_c]*params[nu]*y[E] - params[gamma]*y[I_C]
    yp[H_H] = params[gamma]*y[I_H] - params[delta_h]*y[H_H]
    yp[H_C] = params[gamma]*y[I_C] - params[delta_c]*y[H_C]
    yp[C_C] = params[delta_c]*y[H_C] - params[xi_c]*y[C_C]
    yp[R] = params[gamma]*y[I_R] + params[delta_h]*y[H_H] \
        + params[xi_c]*y[C_C]
    return xp, yp


#########################
## CASE WITH VACCINATION
#########################

def one_step_evol_vax_ct(t, x, confi, vacc, params):    
    rel_R = (params[r]-1)*confi + 1
    xp = np.zeros(x.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - vacc*x[S]
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E]
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*x[I_H] - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*x[I_C] - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*x[I_R] + params[delta_h]*x[H_H]  + params[xi_c]*x[C_C] + vacc*x[S]

    return xp



def one_step_evol_vax_ct_with_grad(t, x, y, z, confi, vacc, week, params):    
    rel_R = (params[r]-1)*confi + 1
    """derivative of vaccine"""
    vaccp = vacc*(1- (vacc/Lambda_v))
    
    xp = np.zeros(x.shape)
    yp = np.zeros(y.shape)
    zp = np.zeros(z.shape)
    
    xp[S] = -rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - vacc*x[S]
    xp[E] = rel_R*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C]) - params[nu]*x[E]
    xp[I_R] = params[p_r]*params[nu]*x[E] - params[gamma]*x[I_R]
    xp[I_H] = params[p_h]*params[nu]*x[E] - params[gamma]*x[I_H]
    xp[I_C] = params[p_c]*params[nu]*x[E] - params[gamma]*x[I_C]
    xp[H_H] = params[gamma]*x[I_H] - params[delta_h]*x[H_H]
    xp[H_C] = params[gamma]*x[I_C] - params[delta_c]*x[H_C]
    xp[C_C] = params[delta_c]*x[H_C] - params[xi_c]*x[C_C]
    xp[R] = params[gamma]*x[I_R] + params[delta_h]*x[H_H] + params[xi_c]*x[C_C] + vacc*x[S]
    """first, the part that does not depend on confi explicitly"""
    yp[S] = -rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - vacc*y[S]
    yp[E] = rel_R*beta(t, params)*(x[S]*(y[I_R] + y[I_H] + y[I_C]) + 
                                    y[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*y[E]
    yp[I_R] = params[p_r]*params[nu]*y[E] - params[gamma]*y[I_R]
    yp[I_H] = params[p_h]*params[nu]*y[E] - params[gamma]*y[I_H]
    yp[I_C] = params[p_c]*params[nu]*y[E] - params[gamma]*y[I_C]
    yp[H_H] = params[gamma]*y[I_H] - params[delta_h]*y[H_H]
    yp[H_C] = params[gamma]*y[I_C] - params[delta_c]*y[H_C]
    yp[C_C] = params[delta_c]*y[H_C] - params[xi_c]*y[C_C]
    yp[R] = params[gamma]*y[I_R] + params[delta_h]*y[H_H]  + params[xi_c]*y[C_C] + vacc*y[S]
    zp[S] = -rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - vacc*z[S]
    zp[E] = rel_R*beta(t, params)*(x[S]*(z[I_R] + z[I_H] + z[I_C]) + 
                                    z[S]*(x[I_R] + x[I_H] + x[I_C])) - params[nu]*z[E]
    zp[I_R] = params[p_r]*params[nu]*z[E] - params[gamma]*z[I_R]
    zp[I_H] = params[p_h]*params[nu]*z[E] - params[gamma]*z[I_H]
    zp[I_C] = params[p_c]*params[nu]*z[E] - params[gamma]*z[I_C]
    zp[H_H] = params[gamma]*z[I_H] - params[delta_h]*z[H_H]
    zp[H_C] = params[gamma]*z[I_C] - params[delta_c]*z[H_C]
    zp[C_C] = params[delta_c]*z[H_C] - params[xi_c]*z[C_C]
    zp[R] = params[gamma]*z[I_R] + params[delta_h]*z[H_H] + params[xi_c]*z[C_C] + vacc*z[S]
    """now, the part with the partial derivative of G^i"""
    zp[S, week] += - vaccp*x[S]
    zp[R, week] += vaccp*x[S]
    
    return xp, yp, zp



#computes the leap of the gradient vector when it crosses
def leap(t, t_0, t_f, x, u_min, u_max, yepa, params):
    laplacian = soft_max_diff(yepa)
    salto = np.zeros(len(yepa))
    for u in range(u_min, u_max + 1):
        for k in range(u + 1):
            salto += laplacian[k, :]*(-1)**(u+1)
            
    salto = salto*(t_f - t_0)
    yp = np.zeros((9, len(yepa)))
    yp[S] = -(params[r]-1)*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*salto
    yp[E] = (params[r]-1)*beta(t, params)*x[S]*(x[I_R] + x[I_H] +x[I_C])*salto
    
    return yp
    



############################################################################
#### GENERAL FUNCTIONS FOR THE EVOLUTION OF COMPARTMENTS
############################################################################

###function to evolve the system without intervention, starting from 
###some initial conditions t_0, and ending at t_f
### N is discrete num of days   
### returns the final occupation of the vector of compartments
    
def future_nat(x_0, t_0, t_f, params):
    N = int(t_f-t_0)
    x = x_0
    ## at time t_0, x= x_!!!
    for k in range(N):
        t = t_0 + k        
        x += one_step_evol(t, x, 0.0, params)
    return x


###returns the occupation of the compartments at time t_0, assuming that 
###the outbreak happened at time t_0-offset
    
def initial_pop(offset, t_0, params):
    ## compartment occupation at the outbreak: 10 exposed people
    x_0 = np.zeros(Num_comp)
    x_0[S] = 1.0 - 10.0/population
    x_0[E] = 10.0/population
    x = future_nat(x_0, t_0 - offset, t_0, params)
    return x


###evolve the compartment vector x for N days
###create a history of the evolution at each time step in a matrix.
def evolve(x_0, t_0, t_f, params, confinement):
    N = int(t_f-t_0)
    x = np.zeros(list(x_0.shape) + [N])
    x[:, 0] = x_0
    for k in range(N - 1):
        t = t_0 + k
        x[:, k + 1] = x[:, k] + one_step_evol(t, x[:, k], confinement(t),  params)
    return x


def evolve_vax(x_0, t_0, t_f, params, confinement, vaccine):
    N = int(t_f-t_0)
    x = np.zeros(list(x_0.shape) + [N])
    x[:, 0] = x_0
    for k in range(N - 1):
        t = t_0 + k
        x[:, k + 1] = x[:, k] + one_step_evol_vax(t, x[:, k], confinement(t), vaccine(t),  params)
    return x

def evolve_vax_epsilon(x_0, t_0, t_f, params, confinement, vaccine, epsilon):
    N = int((t_f-t_0)/epsilon)
    x = np.zeros(list(x_0.shape) + [N])
    x[:, 0] = x_0
    for k in range(N - 1):
        t = t_0 + k*epsilon
        x[:, k + 1] = x[:, k] + epsilon*one_step_evol_vax(t, x[:, k], confinement(t), vaccine(t),  params)
    return x


## Evolve stochastic model with gaussian noise
def evolve_stoch_mod(x_0, t_0, t_f, params, confinement, noise_level):
    N = int(t_f-t_0)
    x = np.zeros(list(x_0.shape) + [N])
    x[:, 0] = x_0
    for k in range(N - 1):
        t = t_0 + k
        ## create noise with gaussian distribution
        noise = np.random.normal(0,1)*noise_level

        x_p =  one_step_evol(t, x[:,k], confinement(t), params)

        noise = x_p[0]*noise
        
        x[:,k + 1]  = x[:,k] + x_p
        ## Act only on the first and second compartments
        x[0, k + 1] = x[0, k + 1] + noise
        x[1, k + 1] = x[1, k + 1] - noise

    return x

## Evolve stochastic model with uniformely distributed noise
def evolve_stoch_mod_unif(x_0, t_0, t_f, params, confinement, noise_level):
    N = int(t_f-t_0)
    x = np.zeros(list(x_0.shape) + [N])
    x[:, 0] = x_0
    for k in range(N - 1):
        t = t_0 + k
        ## create noise with flat distribution in [-noise_level/2, noise_level/2]
        
        noise_v = noise_level*(2*np.random.rand(len(x)-1)-1)
        
        x_p =  one_step_evol(t, x[:,k], confinement(t), params)
        for j in range(len(x)-1):
            noise_v[j] = x_p[j]*noise_v[j]

        noise_v = np.append(noise_v, -np.sum(noise_v))

        x[:, k + 1] = x[:, k] + x_p + noise_v
    return x


###same as above but evolving the system from x_0, t_0 for w weeks only
def evolve_w(t_0, w, x_0, confi, params):
    nd = w*7
    x = np.zeros((len(x_0),nd+1))
    x[:,0] = x_0.copy()
    for i in range(nd):
        t = t_0 + i
        x[:,i+1] = x[:,i] + one_step_evol(t, x[:,i], confi[i//7], params)
    return x


###natural evolution of the compartments from x_0, t_0 or N days 
###without any intervention whatsoever
def evolve_nat(x_0, t_0, t_f, params):
    def confinement(t):
        return 0.0
    return evolve(x_0, t_0, t_f, params, confinement)

###evolves the compartments from x_0, t_0 for N days with a pre-loaded policy
def evolve_with_intervention(poli, x_0, t_0, t_f, params):
    return evolve(x_0, t_0, t_f, params, poli.confinement)

###evolves the compartments from x_0, t_0 for N days with a pre-loaded policy
def evolve_with_intervention_vax(poli, x_0, t_0, t_f, params):
    return evolve_vax(x_0, t_0, t_f, params, poli.confinement, poli.vaccine)


###evolves the compartments from x_0, t_0 for N days with a pre-loaded policy
def evolve_with_intervention_vax_epsilon(poli, x_0, t_0, t_f, params, epsilon):
    return evolve_vax_epsilon(x_0, t_0, t_f, params, poli.confinement, poli.vaccine, epsilon)

###returns the final occupation of the vector of compartments x
###after a weekly policy plan is applied to it
def future_intervention(x_0, t_0, t_f, confi,  params):
    N = int(t_f-t_0)
    x = x_0.copy()
    week = -1
    for k in range(N):
        t = t_0 + k
        if k % 7 ==0:
            week +=1
        x += one_step_evol(t, x, sigmoid(confi[week]),  params)
    return x    

###returns the final occupation of the vector of compartments x
###after natural evolution (no intervention)
def future_nat_ct(x_0, t_0, t_f, params, epsilon=0.2):
    N = int((t_f-t_0)/epsilon)
    x = x_0.copy()
    #at time t_0, x= x_0!!!
    for k in range(N):
        t = t_0 + k*epsilon
        x += epsilon*one_step_evol_ct(t, x, 0.0, params)
    return x


##computes the initial population of the compartments at the beginning
###of the outbreak
def initial_pop_ct(offset, t_0, params, epsilon = 0.2):
    ## compartment occupation at the outbreak: ten exposed people
    x_0 = np.zeros(9)
    x_0[S] = 1.0 - 10.0/population
    x_0[E] = 10.0/population
    x = future_nat_ct(x_0, t_0 - offset, t_0, params, epsilon=epsilon)
    return x


##For the case of a discrete number of lockdowns this function
##returns the first i in the tau vector such that t> cumsum(tau)[i]
def surpassed_tau(t,tau):
    csum = 0
    ind = 1
    for i in range(len(tau)+1):
        if t >= csum:
            csum += tau[i]
        else:
            ind = i
            break
    return ind - 1

##Return confinment or no confinement depending on which interval is surpassed 
def confinement_tau(t,tau):
    i = surpassed_tau(t,tau)
    return int((1 - (-1)**i)//2)



def avg_evol_w(t, w, x_0, confi, params,  uncert, noise_level):

    num_fluct = 100#max(int(100*noise_level), 25)
    nd = 7*w
    x_vec = np.zeros((num_fluct, len(x_0), nd+1))
    st_params = np.zeros((num_fluct, len(params)))

    for i in range(num_fluct):
        st_params[i] = rand_params_singv(params, uncert, noise_level)
        x_vec[i] = evolve_w(t, w, x_0, confi, st_params[i] )
        
    x = np.array(x_vec).mean(axis=0) ##take the average over the evolutions
    xmin = np.array(x_vec).min(axis=0) ##take the min over the evolutions
    xmax = np.array(x_vec).max(axis=0) ##take the max over the evolutions

    return x, xmin, xmax
