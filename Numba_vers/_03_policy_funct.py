import numpy as np
from numba import vectorize, jit, njit, prange, cuda, float32, float64, int32, int64, uint16, uint32, boolean, types
import numba as nb

from _01_params_funct import *


###############################################################################
###############################################################################
#
#            DEFINE PHYSICAL DISTANCING AND DELIBERATE INFECTION VECTORS
#
###############################################################################
###############################################################################

### note that 'confinement' should be interpreted as physical distancing
### note that 'inoculation' should be interpreted as deliberate infection



### continuous physical distancing function
@jit('float64(float64)',nopython=True)
def confinement(confi_el):
    return sigmoid(confi_el)

### continuous vaccination function
@jit('float64(float64)',nopython=True)
def vaccine(vax_el):
    return Lambda_v*sigmoid(vax_el)

### continuous deliberate infection function 
@jit(['float64(float64, boolean)', 'float64(float64, Omitted(True))'])
def infection(inoc_el, ethical=True):
    if ethical == True:
        inoc = 0.
    else:
        inoc = Lambda_i*sigmoid(inoc_el)
    return inoc

### total sum of continous physical distancing in weeks
@jit('float64(float64[:])',nopython=True)
def total_confinement(confi):
    return 7.0*np.sum(sigmoid(confi))

### total sum of vaccination in weeks
@jit('float64(float64[:])',nopython=True)
def total_vaccines(vax):
    return Lambda_v*7.0*np.sum(sigmoid(vax))


### Sampling function for the optimization of lockdown policies (0,1 valued)
### confi vector represents the probability of a lockdown
@jit('float64[:](float64[:])',nopython=True)
def sample_conf(confi_prob):
    ## take a probabilistic policy confi_prob with Nw elements and binarize
    ## it by random sampling 
    confi = np.zeros(len(confi_prob))
    rnd =np.random.rand(len(confi_prob))
    for k in range(len(confi_prob)):
        if confi_prob[k] > rnd[k]:
            confi[k] = 1
    return confi

### take vector of physical distancing probabilities and generate two  
### binary policies, one with confi[j]=0 and one with confi[j]=1
### Nw = number of weeks
### see equation (D5) of paper 
@jit('float64[:](float64[:],int32)',nopython=True)
def sample_conf_butj(confi_prob,j):
    Nw = len(confi_prob)
    confi = np.zeros(2*Nw)
    confi[j] = 0
    rnd =np.random.rand(len(confi_prob))
    for k in range(len(confi_prob)):
        if confi_prob[k] > rnd[k] and k != j:
            confi[k] = 1

    confi[Nw+j] = 1
    rnd =np.random.rand(len(confi_prob))
    for k in range(len(confi_prob)):
        if confi_prob[k] > rnd[k] and k != j:
            confi[Nw+k] = 1

    return confi


###############################################################################
###############################################################################
#
#                        DEFINE POLICY CLASS
#
###############################################################################
###############################################################################

### A class that returns a disease control policy for N days
### where the measure is decided weekly (number of weeks = num_pieces)   
### time is discretised to every day
class Policy():
    def __init__(self, t_0, t_f, ethical = True):
        num_pieces = int((t_f - t_0)/7.0) + 1
        #self.w = torch.randn(num_pieces, 2)/np.sqrt(num_pieces)
        self.confi = 0.0*np.ones(num_pieces) #why??
        self.inoc = np.zeros(num_pieces)
        self.vax = np.zeros(num_pieces)
        self.t_0 = t_0
        self.t_f = t_f
        self.ethical = ethical
        self.num_pieces = num_pieces
    
    def infection(self, t):
        assert t <= self.t_f, "Input time exceeds policy definition."
        ind = int((t - self.t_0)/7.0)
        if self.ethical == True:
            inoc = 0.0
        else:
            inoc = Lambda_i*sigmoid(self.inoc[ind])
        return inoc
    
    def confinement(self, t):
        assert t <= self.t_f, "Input time exceeds policy definition."
        ind = int((t - self.t_0)/7.0)
        return sigmoid(self.confi[ind])
    
    def total_confinement(self):
        return 7.0*sigmoid(self.confi).sum()

    def vaccine(self, t):
        assert t <= self.t_f, "Input time exceeds policy definition."
        ind = int((t - self.t_0)/7.0)
        vax = Lambda_v*sigmoid(self.vax[ind])
        return vax

    def total_vaccines(self):
        return 7.0*Lambda_v*sigmoid(self.vax).sum()



### A class that returns a disease control policy for N days
### where the measure is decided weekly (number of weeks = num_pieces)
### time is continuous (up to epsilon)
class Policy_ct():
    def __init__(self, t_0, t_f, num_pieces):
        num_weeks = int((t_f-t_0)//7) + 1
        self.yepa = np.zeros(num_pieces)
        self.vax = np.zeros(num_weeks)
        self.t_0 = t_0
        self.t_f = t_f
        self.num_pieces = num_pieces
        self.num_weeks = num_weeks
    
    def confinement(self, t):        
        assert t <= self.t_f, "Input time exceeds policy definition."
        tau = (self.t_f - self.t_0)*soft_max(self.yepa)
        
        pieces_greater = np.arange(self.num_pieces)[tau.cumsum() > t - self.t_0]
        ## first piece greater than t - t_0 
        ind = pieces_greater[0]
        return (1 - (-1)**ind)/2
        
        
    def total_confinement(self):
        ## sum of the total physical distancing time 
        tau = (self.t_f - self.t_0)*soft_max(self.yepa)
        
        return tau[1::2].sum()

    def vaccine(self, t):
        assert t <= self.t_f, "Input time exceeds policy definition."
        ind = int((t - self.t_0)/7.0)
        vacc = Lambda_v*sigmoid(self.vax[ind])
        return vacc

    def total_vaccines(self):
        return 7.0*Lambda_v*sigmoid(self.vax).sum()



###############################################################################
###############################################################################
#
#                        SAVE AND LOAD POLICES
#
###############################################################################
###############################################################################


### save policy (discrete time in days)
def save_policy(file_name, poli):
    ## build dictionary 
    np.savez(file_name + ".npz", t_0= poli.t_0, t_f= poli.t_f, 
                 ethical= poli.ethical, confi = poli.confi, inoc = poli.inoc, vax = poli.vax)

###code to load a policy
def load_policy(file_name):
    dic_poli = np.load(file_name + ".npz")
    poli = Policy(dic_poli["t_0"], dic_poli["t_f"], dic_poli["ethical"])
    poli.confi = dic_poli["confi"]
    poli.inoc = dic_poli["inoc"]
    poli.vax = dic_poli["vax"]
    return poli

    
### save a policy with continous time
def save_policy_ct(file_name, poli):
    ## build dictionary 
    np.savez(file_name + ".npz", t_0= poli.t_0, t_f= poli.t_f, 
                 num_pieces = poli.num_pieces, num_weeks = poli.num_weeks, yepa = poli.yepa, vax = poli.vax)

### load a policy with continous time
def load_policy_ct(file_name):
    dic_poli = np.load(file_name + ".npz")
    poli = Policy_ct(dic_poli["t_0"], dic_poli["t_f"], dic_poli["num_pieces"])
    poli.yepa = dic_poli["yepa"]
    poli.vax = dic_poli["vax"]
    return poli
