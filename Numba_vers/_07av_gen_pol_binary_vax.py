import numpy as np
import time


from _01_params_funct import average_values_params, sigmoid
from _02abc_evolution_funct import evolve_nat, initial_pop
from _03_policy_funct import Policy, save_policy, sample_conf
from _05_optimization import optimize_policy_binary_vax
from _06b_actions_cont_phys_dist import action_conf_vax
from _06a_actions_binary import action_conf_binary_vax

from _06a_actions_binary import action_conf_binary_vax_deriv



###############################################################################
###############################################################################
#
#              GENERATE CONTINUOUS PHYSICAL DISTANCING POLICY WITH VACCINATION
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


pol = Policy(t_0, t_f, ethical = True)
num_iter = 100
batch_size = 10
print("Run with ", num_iter," iterations")

## run the optimisation and time it 
tic = time.time()
optimize_policy_binary_vax(action_conf_binary_vax, action_conf_binary_vax_deriv, t_0, t_f, pol.confi, pol.vax,
                           x_0, params = params, lr = 0.01, num_iter=num_iter, batch_size = batch_size)
toc = time.time()
print("Task finished in {} minutes.". format(int((toc-tic)/60)))

save_policy("Pol_vax_binary_"+str(num_iter)+"iter_10changes", pol)
