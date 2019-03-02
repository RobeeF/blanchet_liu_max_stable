# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:34:41 2019

@author: Robin Fuchs
"""

import numpy as np
import math
from simulate_M import algorithm_M, compute_a
import numpy.linalg as nl

#===========================================================
# Estimation part
#===========================================================

def g(n):
    ''' Survival function of L: g(n)=P(L>=n) 
    n (positive int): random variable taking value in N*
    -----------------------------------------------------------------------
    returns (float in (0,1)): the probability that L>=N  
    '''
    return 1/n*np.log(n+np.exp(1)-1)*np.log(np.log(n+np.exp(np.exp(1))-1))

def simulate_L(nb_iter=100, epsilon=10**(-6)):
    ''' Simulate L by Monte Carlo generalized inversion technique and dichotomic search 
    - nb_iter (int): The maximum number of iterations of the dichotomic search algorithm
    - epsilon (small valued float): Stop criterion 
    --------------------------------------------------------------------------
    returns: (positive int): L  
    '''
    u = np.random.uniform(high=1,low=0,size=1)[0]
    INF_search = 0 
    SUP_search = 10**20
    
    L = (SUP_search + INF_search)/2
    while nb_iter>=0: 
        g_n = g(L)
        if (g_n - u <-epsilon/2): # L has to decrease in the next iteration
            SUP_search = L
        elif (g_n - u > epsilon/2): # L has to increase in the next iteration
            INF_search = L
        else: # Convergence was reached
            return int(np.floor(L)) # Maybe ceil or round is more relevant than floor
        nb_iter-=1
        
        L = (INF_search+SUP_search)/2


    raise RuntimeError('Algorithm to find L has failed...')
    

def compute_V_x(x,cov,L):
    ''' Computes the Malliavin-Thalmaier estimator 
    - x (1xd array): The point at which the density has to be evaluated
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    - L (positive int): The number of M estimations used to compute the estimator
    --------------------------------------------------------------------------
    returns: (float): V_x  
    '''
    a = compute_a(cov, 0.05) # Take delta equal to 0.05 
    
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
    
    # Return Sum(W_n - W_{n-1}) for all n in 1,..., L
    return (((W_0_to_L[1:] - W_0_to_L[:-1]))/g_1_to_L).sum()  


def compute_f_hat_b(x, b, cov, conf_lvl=.05):
    '''Estimates the f density at point x with a computational budget of b. 
        Also returns the confidence intervals at 1-conf_lvl %
    - x (1xd array): The point at which the density has to be evaluated
    - b (positive int): The computational budget
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    - conf_lvl (float in (0,1)): The level of the confidence interval
    --------------------------------------------------------------------------
    returns: (float in (0,1)): ^f_b(x)  
    '''
    n = 0
    Tn = 0    
    V_x = []
    
    while Tn<=b:# While the budget is not reached
        L = simulate_L()
        V_x.append(compute_V_x(x,cov,L))
        Tn+=L+n
        n+=1
        
    B = n # B is equal to biggest n such that Tn<=b, i.e. equal to the n at which the previous loop stopped
    f_hat_x_b = np.sum(V_x)/B
    CI = compute_confidence_interval(V_x,f_hat_x_b, b, conf_lvl)
    return f_hat_x_b, CI

#===========================================================
# Confidence interval
#===========================================================
def compute_confidence_interval(V_x, f_hat_x_b, b, conf_lvl): 
    ''' Compute the confidence interval of the estimation of the density
    - V_x (float): The Malliavin-Thalmaier estimator value 
    - f_hat_x_b(float in (0,1)): The estimation of f(x)
    - b (positive int): The computational budget
    - conf_lvl (float in (0,1)): The level of the confidence interval 
    --------------------------------------------------------------------------
    returns: (tuple): The bounds of the interval 
    '''
    s_hat_square = np.square(V_x-f_hat_x_b).mean()
    a_b = np.sqrt(np.log(np.log(np.log(b)))/b)
    quant = np.quantile(V_x,1-conf_lvl/2) # Empirical quantile
    return (f_hat_x_b - quant*np.sqrt(s_hat_square*a_b), f_hat_x_b + quant*np.sqrt(s_hat_square*a_b))
