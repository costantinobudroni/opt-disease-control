import numpy as np
import time


from _01_params_funct import average_values_params
from _02abc_evolution_funct import evolve_nat, initial_pop, sigmoid
from _03_policy_funct import Policy_ct, save_policy_ct, sample_conf
from _05_optimization import optimize_policy_ct
from _06c_actions_cont_time import action_conf_ct, action_conf_deriv_ct

###############################################################################
###############################################################################
#
#              GENERATE CONTINUOUS PHYSICAL DISTANCING POLICY 
#
###############################################################################
###############################################################################



## offset of the outbreak
offset = 30.0
## initial time
t_0 = 60.0
t_f = t_0 + 2*365.0

## initial population
params = average_values_params()
x_0 = initial_pop(offset, t_0, params)
## natural evolution of the disease
x_nat = evolve_nat(x_0, t_0, t_f, params)



## Train continuous-time policy with 15 intervals
num_pieces= 15
pol = Policy_ct(t_0, t_f,num_pieces)
num_iter = 100
print("Actual run with ", num_iter," iterations")
tic = time.time()
optimize_policy_ct(action_conf_ct, action_conf_deriv_ct, t_0, t_f, pol.yepa, x_0,
                             params = params, lr = 0.01, num_iter=num_iter)

toc = time.time()

print("Task finished in {} seconds". format(toc-tic))
save_policy_ct("Pol_ct_"+str(num_iter)+"iter", pol)
