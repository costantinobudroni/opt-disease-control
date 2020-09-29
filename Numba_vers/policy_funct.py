import numpy as np
from numba import vectorize, jit, njit, prange, cuda, float32, float64, int32, int64, uint16, uint32, boolean, types
import numba as nb

from params_funct import *
from evolution_funct import sigmoid

## Substitute policy class with vectors and functions

@jit('float64[:](int32, int32)',nopython=True)
def confi_vector(t_0, t_f):
    num_pieces = int((t_f - t_0)/7.0) + 1
    confi = np.zeros(num_pieces)
    return confi

@jit('float64[:](int32, int32)',nopython=True)
def inoc_vector(t_0, t_f):
    num_pieces = int((t_f - t_0)/7.0) + 1
    inoc = np.zeros(num_pieces)
    return inoc

@jit('float64(float64)',nopython=True)
def confinement(confi_el):
    return sigmoid(confi_el)


@jit(['float64(float64, boolean)', 'float64(float64, Omitted(True))'])
def infection(inoc_el, ethical=True):
    if ethical == True:
        inoc = 0.

    else:
        inoc = Lambda*sigmoid(inoc_el)

    return inoc


@jit('float64(float64[:])',nopython=True)
def total_confinement(confi):
    return 7.0*np.sum(sigmoid(confi))


## Sampling functions for the optimization of lockdown policies (0,1 valued)
## confi vector now represents the probability of a lockdown
@jit('float64[:](float64[:])',nopython=True)
def sample_conf(confi_prob):
    ## take the whole probabilistic policy (Nw elements) and binarize it by sampling 
    confi = np.zeros(len(confi_prob))
    rnd =np.random.rand(len(confi_prob))
    for k in range(len(confi_prob)):
        if confi_prob[k] > rnd[k]:
            confi[k] = 1

    return confi

@jit('float64[:](float64[:],int32)',nopython=True)
def sample_conf_butj(confi_prob,j):
    ## take vector of confinement probabilities and generate two binary policies, one with confi[j]=0 and one with confi[j]=1
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




###This function returns a disease control policy function
class Policy():
    def __init__(self, t_0, t_f, ethical = True):
        num_pieces = int((t_f - t_0)/7.0) + 1
        #self.w = torch.randn(num_pieces, 2)/np.sqrt(num_pieces)
        self.confi = 0.0*np.ones(num_pieces) #why??
        self.inoc = np.zeros(num_pieces)
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
            inoc = Lambda*sigmoid(self.inoc[ind])
        return inoc
    def confinement(self, t):
        assert t <= self.t_f, "Input time exceeds policy definition."
        ind = int((t - self.t_0)/7.0)
        return sigmoid(self.confi[ind])
    def total_confinement(self):
        return 7.0*sigmoid(self.confi).sum()

###Same, but for the case of continuous time parameter (and a fixed number of lockdowns)
class Policy_ct():
    def __init__(self, t_0, t_f, num_pieces):
        self.yepa = np.zeros(num_pieces)
        #self.mu[1::2] = np.ones(len(self.mu[1::2]))
        self.t_0 = t_0
        self.t_f = t_f
        self.num_pieces = num_pieces
    
    def confinement(self, t):        
        assert t <= self.t_f, "Input time exceeds policy definition."
        tau = (self.t_f - self.t_0)*soft_max(self.yepa)
        
        pieces_greater = np.arange(self.num_pieces)[tau.cumsum() > t - self.t_0]
        """first piece greater than t - t_0"""
        ind = pieces_greater[0]
        return (1 - (-1)**ind)/2
        
        
    def total_confinement(self):
        """sum the odd times and return the number"""
        tau = (self.t_f - self.t_0)*soft_max(self.yepa)
        
        return tau[1::2].sum()

###code to save a policy
def save_policy(file_name, poli):
    """build dictionary"""
    np.savez(file_name + ".npz", t_0= poli.t_0, t_f= poli.t_f, ethical= poli.ethical, 
                confi = poli.confi, inoc = poli.inoc)

###code to load a policy
def load_policy(file_name):
    dic_poli = np.load(file_name + ".npz")
    poli = Policy(dic_poli["t_0"], dic_poli["t_f"], dic_poli["ethical"])
    poli.confi = dic_poli["confi"]
    poli.inoc = dic_poli["inoc"]
    return poli

    
###code to save a policy continuous time case
def save_policy_ct(file_name, poli):
    """build dictionary"""
    np.savez(file_name + ".npz", t_0= poli.t_0, t_f= poli.t_f, num_pieces = poli.num_pieces, 
                yepa = poli.yepa)

###code to load a policy continuous time case
def load_policy_ct(file_name):
    dic_poli = np.load(file_name + ".npz")
    poli = Policy(dic_poli["t_0"], dic_poli["t_f"], dic_poli["num_pieces"])
    poli.yepa = dic_poli["yepa"]
    
    return poli
