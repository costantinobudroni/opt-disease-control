import numpy as np
import time


from _01_params_funct import average_values_params, sigmoid
from _02abc_evolution_funct import evolve_nat, initial_pop
from _03_policy_funct import Policy, save_policy, sample_conf
from _05_optimization import optimize_policy, optimize_policy_stoch, optimize_stoch_decr_uncert, optimize_policy_stoch_evo
from _06b_actions_cont_phys_dist import action_conf, action_conf_deriv, action_conf_deriv_stoch, action_conf_deriv_stoch_previous_policy, action_conf_deriv_stoch_evo, action_conf_avg_stoch_evo, action_conf_stoch_evo



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


pol = Policy(t_0, t_f, ethical = True)
num_iter = 100
print("Run with ", num_iter," iterations")

## run the optimisation and time it 
tic = time.time()
optimize_policy(action_conf, action_conf_deriv, t_0, t_f, pol.confi,
                     x_0, params = params, lr = 0.01, num_iter=num_iter)
toc = time.time()
print("Task finished in {} minutes.". format(int((toc-tic)/60)))

save_policy("Pol_cpd_det_"+str(num_iter)+"iter", pol)


###############################################################################
#                       STOCHASTIC PARAMETERS
###############################################################################

#Reset policy
pol = Policy(t_0, t_f, ethical = True)
num_iter = 100

noise_level = [0.05, 0.25 ]
labels = ['05', '25' ]

for k in range(len(noise_level)):
    ## Reset the policy to the initial conditions 
    pol = Policy(t_0, t_f, ethical = True)
    num_iter = 100

    print("Case of stochastic parameters: noise_level =", noise_level[k], " Number of iterations = ", num_iter)
    tic = time.time()
    optimize_policy_stoch(action_conf, action_conf_deriv_stoch, t_0, t_f, pol.confi,
                               x_0,params = params, lr = 0.01, num_iter = num_iter,
                               num_fluct = int(256*noise_level[k]*4), noise_level=noise_level[k] )
    toc = time.time()
    print("Task finished in {} minutes.". format(int((toc-tic)/60)))
    #print("Task finished in {} seconds". format(toc-tic))

    save_policy("Pol_cpd_stoch_noise"+labels[k]+"_"+str(num_iter)+"iter", pol)



###############################################################################
#             STOCHASTIC PARAMETERS WITH DECREASING NOISE 
###############################################################################

#Reset policy
pol = Policy(t_0, t_f, ethical = True)
num_iter = 100

print("Case of stochastic parameters with decreasing noise: noise_level = 0.25", " Number of iterations = ", num_iter)
tic = time.time()
pol.confi = optimize_stoch_decr_uncert(action_conf, action_conf_deriv_stoch_previous_policy, t_0, t_f, pol.confi, x_0,
                                       lr = 0.01, num_iter=num_iter, num_fluct=50, initial_noise_level=0.25)

toc = time.time()
print("Task finished in {} minutes.". format(int((toc-tic)/60)))


save_policy("Pol_cpd_stoch_decr_noise_25_"+str(num_iter)+"iter", pol)



###############################################################################
#                       STOCHASTIC EVOLUTION
###############################################################################

#Reset policy
pol = Policy(t_0, t_f, ethical = True)
num_iter = 100

noise_level = [0.05,0.25 ]
labels =['05', '25' ]

for k in range(len(noise_level)):
    ## Reset the policy to the initial conditions 
    #pol = Policy(t_0, t_f, ethical = True)



    print("Stochastic evolution: case of noise_level =", noise_level[k], " Number of iterations = ", num_iter)
    tic = time.time()
    optimize_policy_stoch_evo(action_conf_stoch_evo, action_conf_avg_stoch_evo, t_0, t_f, pol.confi,
                               x_0,params = params, lr = 0.01, num_iter = num_iter,
                               num_fluct =  int(256*noise_level[k]*4) , noise_level=noise_level[k] )

    toc = time.time()
    print("Task finished in {} minutes.". format(int((toc-tic)/60)))
    #print("Task finished in {} seconds". format(toc-tic))

    save_policy("Pol_cpd_gauss_stoch_evo"+labels[k]+"_"+str(num_iter)+"iter_low_fluct", pol)


