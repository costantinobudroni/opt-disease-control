import numpy as np
import time


from _01_params_funct import average_values_params, sigmoid
from _02abc_evolution_funct import evolve_nat, initial_pop
from _03_policy_funct import Policy_ct, save_policy_ct
from _05_optimization import optimize_policy_vax_ct
from _06c_actions_cont_time import action_conf_vax_ct, action_conf_vax_deriv_ct



###############################################################################
###############################################################################
#
#              GENERATE CONTINUOUS TIME POLICY WITH VACCINATION
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

num_lockdowns = 10
pol = Policy_ct(t_0, t_f, num_lockdowns)
num_iter = 100
print("Run with ", num_iter," iterations")

## run the optimisation and time it 
tic = time.time()
optimize_policy_vax_ct(action_conf_vax_ct, action_conf_vax_deriv_ct, t_0, t_f, pol.yepa, pol.vax,
                           x_0, params = params, lr = 0.01, num_iter=num_iter)
toc = time.time()
print("Task finished in {} minutes.". format(int((toc-tic)/60)))

save_policy_ct("Pol_novax_ct"+str(num_iter)+"iter_"+str(num_lockdowns)+"lockdowns", pol)


