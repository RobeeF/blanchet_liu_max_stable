# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:34:41 2019

@author: robin
"""

import numpy as np
import math
from simulate_M import algorithm_M, compute_a
import numpy.linalg as nl

#===========================================================
# Estimation part
#===========================================================

def g(n):
    return 1/n*np.log(n+np.exp(1)-1)*np.log(np.log(n+np.exp(np.exp(1))-1))

def simulate_L(nb_iter=100, epsilon=10**(-6)):
    ''' Simulate L thanks to MC generalized inversion technique'''
    u = np.random.uniform(high=1,low=0,size=1)[0]
    INF_search = 0 
    SUP_search = 10**20
    
    L = (SUP_search + INF_search)/2
    while nb_iter>=0: 
        g_n = g(L)
        if (g_n - u <-epsilon/2): # a in [-inf,-epsilon/2]
            SUP_search = L
        elif (g_n - u > epsilon/2): # a in [epsilon/2, inf]
            INF_search = L
        else: #a in [-epsilon/2,epsilon/2]: convergence
            return int(np.floor(L))
        nb_iter-=1
        
        L = (INF_search+SUP_search)/2


    raise RuntimeError('Algorithm to find L has failed...')
    

def compute_V_x(x,cov,L):
    a = compute_a(cov, 0.05) # Take delta equal to 0.05 while it has no influence
    
    d = cov.shape[0]
    w_d= math.pi**(float(d)/2)/math.gamma(float(d)/2 + 1) # Sphere of a d-dimensional ball
    delta_1_to_L = 1/np.log(np.log(np.log(np.arange(1,L+1)+np.exp(np.exp(1))))) # Sequence of perturbations from 1 to L
    g_1_to_L = g(np.arange(1,L+1)) # Sequences of g from 1 to L
    
    M_minus_x_L = [] # Will store the sequence of M^(i)-x for all i in 1,...,L 
    numerator_L = [] # Will store the sequence of scalar products in the numerator for all i in 1,...,L 
    
    inv_cov = np.linalg.inv(cov)

    for i in range(L):
        print(i,' eme simulation')
        M,X,N = algorithm_M(a, cov)
        M_minus_x = M - x
        M_minus_x_L.append(M_minus_x)
        numerator_L.append(np.dot(np.dot(X,inv_cov),M_minus_x.T).sum())

    M_minus_x_L = np.stack(M_minus_x_L)
    M_minus_x_norm = nl.norm(M_minus_x_L, ord=2, axis=1)    
    numerator_L = np.array(numerator_L)
    

    regular_denom_term_L = d*w_d*(M_minus_x_norm)**d # The left part of the denominator

    perturbation_denom_term_L = M_minus_x_norm*delta_1_to_L # The perturbation decreases with L
    denominator_L = regular_denom_term_L + perturbation_denom_term_L
    
    # Compute the W_n sequence from 1 to L
    W_1_to_L = numerator_L/denominator_L 
    # As W_0=0 we insert it at the beginning of the sequence
    W_0_to_L = np.insert(W_1_to_L,0,0)
    
    # Return W_n - W_{n-1} for all n in 1,..., L
    return (((W_0_to_L[1:] - W_0_to_L[:-1]))/g_1_to_L).sum()  


def compute_f_hat_b(x, b, cov, conf_lvl=.05):
    n = 0
    Tn = 0    
    V_x = []
    
    while Tn<=b:
        L = simulate_L()
        V_x.append(compute_V_x(x,cov,L))
        Tn+=L+n
        n+=1
    B = n
    f_hat_x_b = np.sum(V_x)/B
    CI = compute_confidence_interval(V_x,f_hat_x_b, b, conf_lvl)
    return f_hat_x_b, CI

#===========================================================
# Confidence interval
#===========================================================
def compute_confidence_interval(V_x, f_hat_x_b, b, conf_lvl): 
    s_hat_square = np.square(V_x-f_hat_x_b).mean()
    a_b = np.sqrt(np.log(np.log(np.log(b)))/b)
    quant = np.quantile(V_x,1-conf_lvl/2) # Empirical quantile
    return [f_hat_x_b - quant*np.sqrt(s_hat_square*a_b), f_hat_x_b + quant*np.sqrt(s_hat_square*a_b) ]
