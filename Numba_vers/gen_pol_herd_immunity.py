import numpy as np
import time
from optimization import optimize_policy_del_inf
from actions_herd_imm import action_herd, action_herd_deriv
from policy_funct import Policy, save_policy
from params_funct import average_values_params
from evolution_funct import evolve_nat, initial_pop


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


pol = Policy(t_0, t_f, ethical = False)
num_iter = 300
print("Run with ", num_iter," iterations")
tic = time.time()
optimize_policy_del_inf(action_herd, action_herd_deriv, t_0, t_f, pol.confi, pol.inoc,
                     x_0, params = params, lr = 0.01, num_iter=num_iter)

toc = time.time()
print("Task finished in {} minutes.". format(int((toc-tic)/60)))
#print("Task finished in {} seconds". format(toc-tic))
save_policy("Pol_herd_imm_"+str(num_iter)+"iter", pol)


