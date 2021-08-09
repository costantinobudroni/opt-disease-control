import numpy as np
import time

from _01_params_funct import average_values_params, sigmoid
from _02abc_evolution_funct import evolve_nat, initial_pop
from _03_policy_funct import Policy, save_policy, sample_conf
from _05_optimization import optimize_policy_binary
from _06a_actions_binary import action_conf_binary, action_conf_binary_deriv, action_conf_probabilistic


###############################################################################
###############################################################################
#
#                       GENERATE A BINARY POLICY 
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
print("Actual run with ", num_iter," iterations")
tic = time.time()

## run the optimization 
optimize_policy_binary(action_conf_probabilistic, action_conf_binary_deriv , t_0, t_f, pol.confi,
                       x_0, params = params, lr = 0.01, batch_size = 10, num_iter=num_iter)

toc = time.time()
print("Task finished in {} seconds". format(toc-tic))
save_policy("Pol_bin_prob_"+str(num_iter)+"iter", pol)

minloss = 500# msb.action_conf_probabilistic(t_0, t_f, pol.confi, pol.inoc, x_0, params)
print("minloss from optimization", minloss)
loss = 0
for _ in range(100):
    confi_bin = sample_conf(sigmoid(pol.confi))
    newloss = action_conf_binary(t_0, t_f, confi_bin, x_0, params)
    if newloss < minloss:
        minloss = newloss
        opt_conf = confi_bin

    loss += newloss

avg_loss = loss /100
print("Average loss over 100 iterations with probabilistic binary policy:", avg_loss)
print("Minimal loss:", minloss)
print("Optimal confinement binary policy", opt_conf)
