import numpy as np


####################################################################################
####################################################################################
#
#                             MODEL PARAMETERS
#
####################################################################################
####################################################################################


###order numbers for the parameters
p_h = 0
p_c = 1
p_r = 2
gamma = 3
delta_h = 4
delta_c = 5
xi_c = 6
basic_rep_num = 7
Delta = 8
phi = 9
nu = 10
r = 11

##critical care capacity (for some average EU country)
C_bound = 9.5/10**5

##target care capacity
C_target = 0.5*C_bound

##step size
epsilon = 1.0

##population (for some average EU country)
population = 47.0*10**6

##maximum rate of deliberate infection
Lambda_i = 10.0**4/population

##maximum rate of vaccination
Lambda_v = 0.5*10.0**5/population

##maximum fraction of the population that gets a vaccine
V=1/3

##order numbers for compartments
S = 0
E = 1
I_R = 2
I_H = 3
I_C = 4
H_H = 5
H_C = 6
C_C = 7
R = 8
Ep = 9
I_Rp = 10
I_Hp = 11 
I_Cp = 12

### Set the number of compartments to 9 for simulations not involving deliberate infection
### Set it to 13 only for the herd immunity with deliberate infection simulations
### Num_comp = 9

### Lagrangian parameter to impose the constraint on critical care capacity
mu = 100/C_bound
### Lagrangian parameter to impose the constraint on maximum number vaccines
mu_v = 10**4


## Function to compute the transmission rate at each instant t
def beta(t, params):
    aux = params[gamma]*params[basic_rep_num]*((1.0 + params[Delta])/2.0 + 
                                               (1.0-params[Delta])*np.cos(
                                                   2*np.pi*(t + params[phi]*7.0)/(7*52.0))/2.0)
    return aux


###############################################################################
###############################################################################
#
#                       RANDOM PARAMETERS SAMPLING FUNCTIONS
#
###############################################################################
###############################################################################


# define ranges for each parameter, the upper and lower
# values are estimated from literature
def ranges_params():
    params = np.zeros((12, 2))
    params[p_h, :] = 0.0308
    params[p_c, :] = 0.0132
    params[p_r, :] = 1.0 - params[p_h, :] - params[p_c, :]
    params[gamma, :] = 1/5
    params[delta_h, :] = 1/8
    params[delta_c, :] = 1/6
    params[xi_c, :] = 1/10
    params[basic_rep_num, :] = [2.0, 2.5]
    params[Delta, :] = [0.7, 1.0]
    params[phi, :] = -3.8
    params[nu, :] = 1/4.6 
    params[r, :] = [0.0, 0.6]
    return params
    

### compute the average of the parameter ranges
def average_values_params():
    params = ranges_params().mean(1)
    return params

def uncertainty_interval():
    rangos = ranges_params()
    uncert = (rangos[:, 0] - rangos[:, 1])/2 ## why half?
    return uncert


## Generates random deviations with respect to each average value, depending on 
## the noise_level which is between 0 and 1
    

def rand_params_singv(aver, uncert, noise_level):
    return noise_level*(2*np.random.rand(len(aver))-1)*uncert + aver



## critical number of susceptible for herd immunity
params_test = average_values_params()
critical_susc = 1/params_test[basic_rep_num]


###############################################################################
###############################################################################
#
#                            USEFUL FUNCTIONS
#
###############################################################################
###############################################################################


def sigmoid(x):
    return 1/(np.exp(-x) +1)


def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))


def relu(x):
    if x <= 0.:
        rel = 0.
    else :
        rel = x
    return rel


def heaviside(x):
    if x <= 0.:
        hvs = 0.
    else :
        hvs = 1
    return hvs



def soft_max(x):
    y = np.exp(x)
    return y/np.sum(y)



def soft_max_diff(x):
    y = soft_max(x)
    d= len(x)
    yy = np.zeros((d,d))
    for i in range(d):
        for j in range(d):
            yy[i,j]= y[i]*y[j]
    return np.diag(y) - yy






