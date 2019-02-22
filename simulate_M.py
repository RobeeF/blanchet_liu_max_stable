# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:32:59 2019

@author: robin
"""

from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad
from numpy.random import multinomial
import numpy as np
import numpy.linalg as nl


#===========================================================
# Gaussian random fields simulation
#===========================================================

def simul_Xk(cov, size=1):
    ''' Multivariate centered of step dt=1 brownian motion starting at (0,...,0) '''
    # Generate the sequence of multivariate_gaussian of size "size"
    r = multivariate_normal.rvs(cov=cov, size=size)
    if size>1:
        return np.cumsum(r, axis=-1)
    else:  
        return r

#===========================================================
# Compute useful constants
#===========================================================
    
def compute_theta(Ao=0, LAMBDA=1, gamma=.5, nb_iter=10, epsilon=10**(-6)): # Not working
    return 0.6


def compute_Na(d, cov, a, A1, X1, gamma=.5):     
    if A1>0:
        return int(np.ceil(np.exp((np.log(A1/gamma) + nl.norm(X1, ord=np.inf))/(1-a)))) # Return the next n such that the condition is verified
    else: 
        return 0

def compute_no(a, bar_sigma, d, nb_iter=1000, epsilon=10**(-15)): # Change bar_sigma with cov
    INF_search = 0 
    SUP_search = 10**20
    
    bar_sigma_a_ratio= bar_sigma/a
    no = (SUP_search + INF_search)/2

    bound = .5*np.sqrt(np.pi/2)*norm.pdf(bar_sigma_a_ratio)/bar_sigma_a_ratio

    while nb_iter>=0:    

        cond = d*(1-norm.cdf(a*np.log(no/bar_sigma)- bar_sigma_a_ratio))
        
            
        if (cond - bound <-epsilon/2): # a in [-inf,-epsilon/2]
            SUP_search = no
        elif (cond - bound > epsilon/2): # a in [epsilon/2, inf]
            INF_search = no
        else: #a in [-epsilon/2,epsilon/2]: convergence
            return int(np.floor(no))
        nb_iter-=1
        
        no = (INF_search+SUP_search)/2

    raise RuntimeError('Algorithm to find no has failed...')



def compute_K(a, no, bar_sigma):
    u = np.random.uniform(high=1,low=0,size=1)[0]
    bar_sigma_a_ratio= bar_sigma/a

    return int(np.ceil(
            np.exp(bar_sigma_a_ratio**2 +
                   bar_sigma_a_ratio*(norm.ppf(1-u*(1-norm.cdf(a*np.log(no)/bar_sigma - bar_sigma_a_ratio))))) 
                   -no)) # A checker : 1 pas dans la ppf
    
def gauss_of_log(s,a, no, bar_sigma):
    return norm.pdf(a*np.log(no+s)/bar_sigma)


def compute_go_x(k,a,no,bar_sigma):
    return quad(gauss_of_log,k-1,k,args=(a, no, bar_sigma))[0]/quad(gauss_of_log,0,np.inf,args=(a, no, bar_sigma))[0]


def compute_a(cov, delta, Ao=0, LAMBDA=1, gamma=.5, epsilon=.001, nb_iter=500):
    INF_search = 0 
    SUP_search = 1
    NB_SAMPLE = 10000 # Nb of samples used to estimate the expectation

    d = cov.shape[1]
    a = (SUP_search + INF_search)/2
    bar_sigma = np.sqrt(max(np.diag(cov)))

        
    X1 = multivariate_normal.rvs(cov=cov, size=NB_SAMPLE)  # Correct the typo in the paper. Compute the norm of X_1 and not X as in [10]  
    X1_norm_max = np.max(np.abs(X1), axis=1) # Get the max over the d dimensions for each generated sample
    
    A1 = Ao + np.random.poisson(LAMBDA, size=(NB_SAMPLE)) 
    
    while nb_iter>=0:    

        right_hand_side = np.mean(A1*np.exp(X1_norm_max)/gamma,axis=0)**(1/(1-a))
    
        bar_sigma_a_ratio= bar_sigma/a
        left_hand_side = np.exp(bar_sigma_a_ratio*
                                norm.ppf(1-delta*np.sqrt(2*np.pi)*norm.pdf(bar_sigma_a_ratio)/(d*bar_sigma_a_ratio)) 
                                + bar_sigma_a_ratio**2) 
            
        if (left_hand_side - right_hand_side <-epsilon/2): # if a in [-inf,-epsilon/2]
            SUP_search = a
        elif (left_hand_side - right_hand_side>epsilon/2): # if a in [epsilon/2, inf]
            INF_search = a
        else: #a in [-epsilon/2,epsilon/2]: convergence
            return a
            #return a
        nb_iter-=1
        
        a = (INF_search+SUP_search)/2
    raise RuntimeError('Algorithm to find a has failed...')

#===========================================================
# Simulate the Poisson Arrival Process
#===========================================================

def simulate_An(length, Ao=0, LAMBDA=1):
    ''' Returns A1, ..., An and not Ao '''
    return  Ao + np.random.poisson(LAMBDA, size=length).cumsum() 


def simulate_S(length=500, So=0, LAMBDA=1, gamma=.5):
    ''' Returns S1, ..., An and not So '''
    An = simulate_An(length=length,Ao=-So,LAMBDA=LAMBDA) # Compute An
    return (gamma*np.arange(1,length+1) - An) # Check right shape


def compute_NS(S):
    xi_plus = [0]
    xi_minus = [0] # In case of empty sequence returns that NS=0 which is coherent
    inf_not_reached = True
    i=1
    while inf_not_reached: 
        if any(S[xi_plus[i-1]:]<0) and np.isinf(xi_plus[i-1])==False: # If there exist some negative Si for i in S_{x+_{n}}, S_{x+_{n+1}},  S_{x+_{n+2}}, ...
            xi_minus.append(np.argmax(S[xi_plus[i-1]:]<0)+xi_plus[i-1])
        else: # Else x- is infinite
            xi_minus.append(np.inf)
            inf_not_reached = False
        
        if any(S[xi_minus[i]:]>=0) and np.isinf(xi_minus[i])==False:
            xi_plus.append(np.argmax(S[xi_minus[i]:]>=0)+xi_minus[i])
        else:
            xi_plus.append(np.inf)
            inf_not_reached = False
        i+=1 

    NS = xi_minus[-1] # If the biggest xi_minus is inf we take the previous one
    NS = xi_minus[-2] if np.isinf(NS) else NS
    return NS


def sample_downcrossing(x):
    Sn = simulate_S(So=x, LAMBDA=1)
    tau_minus = np.argmax(Sn<0) 
    return Sn[:tau_minus+1].tolist()

def sample_upcrossing(x):
    theta = compute_theta(x)
    Sn = simulate_S(So=x, LAMBDA=1-theta)
    
    S = Sn[:np.argmax(Sn>=0)+1] 
    u = np.random.uniform(high=1,low=0,size=1)[0]
    
    if u<np.exp(-theta*(S[-1]-x)):
        return S.tolist()
    else:
        return 'degenerated'


def algorithm_S_NA(): 
    S = [0.]
    degenerated = False
    while degenerated!=True:
        down_crossing_segment = sample_downcrossing(S[-1])
        S = S + down_crossing_segment
        up_crossing_segment = sample_upcrossing(S[-1])
        
        if up_crossing_segment!='degenerated':
            S = S + up_crossing_segment
        else:
            degenerated=True
    return S


def sample_without_record(x,l):
    S = simulate_S(length=l, So=0, LAMBDA=1) # Doit en prendre l parmi n ??
    while (np.max(S)>=0) or (sample_upcrossing(S[-1])!='degenerated'): # Triggers a useless warning
        S = simulate_S(length=l, So=x, LAMBDA=1)
    return S.tolist()


def algorithm_S_NA_to_N(S,l):
    if l>0:
        S = S + sample_without_record(S[-1],l)
    return S


#===========================================================
# Asymptotic gaussian field simulation
#===========================================================

def modified_bernoulli(p=.5):
    ''' Bernoulli that returns -1 or 1 instead of 0 or 1 with probability p'''
    x = np.random.binomial(n=1,p=0.5)
    x = x if x>0 else -1
    return x

def conditioned_sampleX(a,n, cov): 
    alogn = a*np.log(n)
    
    location_weights = 2*(1- np.array([norm.cdf(alogn, loc=0, scale=sig) for sig in np.sqrt(np.diag(cov))]))
    tv = multinomial(n=1, pvals=location_weights/location_weights.sum()) 
    tv = np.where(tv==1)[0][0]
    
    u = np.random.uniform(high=1,low=0,size=1)[0]
    j = modified_bernoulli()
    
    sigma_tv = np.sqrt(cov[tv,tv])

    X_tv = sigma_tv*j*norm.ppf(u+(1-u)*norm.cdf(alogn/sigma_tv))
    Y =  simul_Xk(cov) # Why don't we take X_{K-1} to start the sequence ?

    return Y - cov[:,tv]*Y[tv]/cov[tv,tv] + X_tv 


def sample_single_record(a,no,n1, cov):  # Does not depend on delta.. 
    bar_sigma = np.sqrt(np.max(np.diag(cov))) 

    K = compute_K(a,no, bar_sigma)

    X = simul_Xk(cov=cov, size=K-1) 
    X_K  = conditioned_sampleX(a,n1+K,cov)
    if K>2:    
        X = np.append(X.tolist(), [X_K.tolist()], axis=0) 
    elif K==2:
        X = np.stack([X,X_K])
    else:
        X = np.array([X_K])
    
    u = np.random.uniform(high=1,low=0,size=1)[0]

    cond1 = all([np.max(np.abs(X[k,:]))<=a*np.log(n1+k+1) for k in range(0,K-1)]) 
        
    theoretical_proba = 2*(1- np.array([norm.cdf(a*np.log(n1+K), loc=0, scale=sig) for sig in np.sqrt(np.diag(cov))])).sum()
    empirical_proba = np.sum(abs(X_K)>a*np.log(n1+K))
    
    cond2 = u*compute_go_x(K,a,no,bar_sigma) < theoretical_proba/empirical_proba # Not sure for cond2 that just inversing the ratio is ok

    if cond1 and cond2:
        return X
    else: 
        return 'degenerated'
    
    
def sample_without_recordX(a, n1,l, cov): # Dimension problem ? Break the dynamic ? 
    X = simul_Xk(cov=cov, size=l) 
    while (np.max(np.abs(X),axis=1) - a*np.log(n1+np.arange(1,l+1))>0).any(): # Dimension problem max for several dimension..
        X = simul_Xk(cov=cov, size=l) 
    return X


def algorithm_X_NX(a, cov):
    d = cov.shape[1]
    bar_sigma = np.sqrt(np.max(np.diag(cov)))
    no = compute_no(a, bar_sigma, d)
    eta = no
    X = simul_Xk(cov=cov, size=eta) 
    segment = True # Sale, Boucle do while ?
    
    while segment!= 'degenerated': # Boucle while+condition a pas trop de sens
        segment = sample_single_record(a,no,eta, cov)
        if segment!='degenerated':
            X = np.append(X.tolist(), segment.tolist(), axis=0)
            eta = len(X)
    return X


def algorithm_X_NX_to_N(a,X,l, cov):
    eta = X.shape[0]
    if l>0: 
        X = np.append(X.tolist(), sample_without_recordX(a,eta, l, cov).reshape(-1,X.shape[1]).tolist(), axis=0)
        return X

    else:
        raise ValueError('Please enter a positive integer value for l')

def algorithm_M(a, cov, gamma=.5, return_moments=False):
    S =  algorithm_S_NA() 
    A = gamma*np.arange(1,len(S)+1) - S
    d = cov.shape[0]
    X = algorithm_X_NX(a, cov).reshape(-1,d)
    
    Na, N_X, N_A = compute_Na(d, cov, a, A[0], X[0,:], gamma), len(X), len(A)
    N = max(Na, N_X, N_A)
    
    if N>N_A:
        S =algorithm_S_NA_to_N(S,N-N_A) 
        A = gamma*np.arange(1,len(S)+1) - S
    if N>N_X:
        X = algorithm_X_NX_to_N(a,X,N-N_X, cov)

    M_t_1_n = -np.log(np.stack([A]*X.shape[1]).T) + X
    
    if return_moments:
        return np.max(M_t_1_n, axis=0), X, N, Na, N_X, N_A 
    else:
        return np.max(M_t_1_n, axis=0), X, N 