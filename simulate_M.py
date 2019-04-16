# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:32:59 2019

@author: Robin Fuchs
"""

from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad
from numpy.random import multinomial
import numpy as np
import numpy.linalg as nl


#===========================================================
# Compute useful constants and utilities
#===========================================================
    
def compute_theta(Ao=0, LAMBDA=1, gamma=.5, nb_iter=100, epsilon=10**(-15)): # Not working
    ''' Compute the Cramer's root that will be used to simulate a Poisson process with unit rate 1+theta
   - Ao (int): The starting point of the Poisson process
   - LAMBDA (float): The Poisson process rate
   - gamma (float in (0,1)): parameter of the random Walk chosen such that gamma < E(A_1)
   - nb_iter (int): The maximum number of iterations of the dichotomic search algorithm
   - epsilon (small valued float): Stop criterion
   --------------------------------------------------------------------------
    returns: (float in [0,1]): Theta  
    '''
    
    INF_search = 0 
    SUP_search = 100
    NB_SAMPLE = 10000
    
    theta = (SUP_search + INF_search)/2
    
    while nb_iter>=0:  
        #print("theta: ",theta)
        A1 = Ao + np.random.poisson(1, size=(NB_SAMPLE)) 
        S1 = -A1 + gamma 
        
        cond = np.abs(np.exp((theta*S1)).mean(axis=0)-1) # Approximate the expectation by a mean over NB_SAMPLE samples

        if (cond-1 <-epsilon/2): # E(exp(theta*S1))-1 in [-inf,-epsilon/2]
            INF_search = theta
        elif (cond-1 > epsilon/2): # E(exp(theta*S1))-1 in [epsilon/2, inf]
            SUP_search = theta
        else: #a in [-epsilon/2,epsilon/2]: convergence
            return theta
        nb_iter-=1
    
        theta = (INF_search+SUP_search)/2
    return theta

def compute_a(cov, delta, Ao=0, LAMBDA=1, gamma=.5, nb_iter=500, epsilon=10**(-3)):
    ''' Compute the parameter value of a, as in the paper page 18 thanks to a dichotomic search algorithm
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    - delta (float in (0,1)): Some unknown parameter from the paper [10]
    - Ao (int): The starting point of the Poisson process
    - LAMBDA (float): The Poisson process rate
    - gamma (float in (0,1)): parameter of the random Walk chosen such that gamma < E(A_1)
    - nb_iter (int): The maximum number of iterations of the dichotomic search algorithm
    - epsilon (small valued float): Stop criterion 
    --------------------------------------------------------------------------
    returns: (float in (0,1)): a  
    '''
    
    INF_search = 0 
    SUP_search = 1
    NB_SAMPLE = 10000 # Nb of samples used to estimate the expectation

    d = cov.shape[1]
    a = (SUP_search + INF_search)/2
    bar_sigma = np.sqrt(max(np.diag(cov)))

         
    X1 = multivariate_normal.rvs(cov=cov, size=NB_SAMPLE)  # Correct the typo in the paper. Compute as in [10] ||X_1||_inf and not ||X||_inf  
    X1_norm_max = nl.norm(X1, ord=np.inf,axis=1) 
    
    A1 = Ao + np.random.poisson(LAMBDA, size=(NB_SAMPLE)) # Simulate NB_SAMPLE A1 realisation
    
    while nb_iter>=0:    
        right_hand_side = np.mean(A1*np.exp(X1_norm_max)/gamma,axis=0)**(1/(1-a)) # Right hand side of the equation page 18
    
        bar_sigma_a_ratio= bar_sigma/a
        left_hand_side = np.exp(bar_sigma_a_ratio*
                                norm.ppf(1-delta*np.sqrt(2*np.pi)*norm.pdf(bar_sigma_a_ratio)/(d*bar_sigma_a_ratio)) 
                                + bar_sigma_a_ratio**2) 
            
        if (left_hand_side - right_hand_side <-epsilon/2): # a must decrease at the next iteration
            SUP_search = a
        elif (left_hand_side - right_hand_side>epsilon/2): # a must increase at the next iteration
            INF_search = a
        else: # Convergence was reached
            return a
        nb_iter-=1
        
        a = (INF_search+SUP_search)/2
    raise RuntimeError('Algorithm to find a has failed...')

def compute_Na(d, cov, a, A1, X1, gamma=.5):     
    ''' Compute the random time Na such that for n>Na: n*gamma > A1*n^a*exp(||X1||_inf)
    - d (int) The dimension of the multivariate vector of random fields
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - A1 (int): The value at time n=1 of the Poisson process
    - X1 (1xd array): The value at time n=1 of X_k
    - gamma (float in (0,1)): parameter of the random Walk chosen such that gamma < E(A_1)
    --------------------------------------------------------------------------
    returns: (int): Na
    '''

    if A1>0:
        return int(np.ceil(np.exp((np.log(A1/gamma) + nl.norm(X1, ord=np.inf))/(1-a)))) 
    else: 
        return 0

def compute_no(a, bar_sigma, d, nb_iter=1000, epsilon=10**(-15)): # Change bar_sigma with cov
    ''' Compute the first record breaking time no with a dichotomic search algorithm
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - bar_sigma (positive float): The biggest variance term of the variance-covariance matrix
    - d (int) The dimension of the multivariate field
    - nb_iter (int): The maximum number of iterations of the dichotomic search algorithm
    - epsilon (small valued float): Stop criterion
    --------------------------------------------------------------------------
    returns: (int): no
    '''
    
    INF_search = 0 
    SUP_search = 10**20 # Start very high
    
    bar_sigma_a_ratio= bar_sigma/a
    no = (SUP_search + INF_search)/2

    bound = .5*np.sqrt(np.pi/2)*norm.pdf(bar_sigma_a_ratio)/bar_sigma_a_ratio # Right hand side of the equation

    while nb_iter>=0:    

        cond = d*(1-norm.cdf(a*np.log(no/bar_sigma)- bar_sigma_a_ratio)) # Left-hand side of the equation
            
        if (cond - bound <-epsilon/2): # no has to decrease at the next iteration
            SUP_search = no
        elif (cond - bound > epsilon/2): # no has to increase at the next iteration
            INF_search = no
        else: # Convergence was reached
            return int(np.floor(no)) # Low-rounding to be sure that cond <= bond
        nb_iter-=1
        
        no = (INF_search+SUP_search)/2

    raise RuntimeError('Algorithm to find no has failed...')



def compute_K(a, no, bar_sigma):
    ''' Compute the random time K as in the paper page 19
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - no (positive int): The first record-breaking time
    - bar_sigma (positive float): The biggest variance term of the variance-covariance matrix
    --------------------------------------------------------------------------
    returns: (int): K    
    '''
    u = np.random.uniform(high=1,low=0,size=1)[0]
    bar_sigma_a_ratio= bar_sigma/a

    return int(np.ceil(
            np.exp(bar_sigma_a_ratio**2 +
                   bar_sigma_a_ratio*(norm.ppf(1-u*(1-norm.cdf(a*np.log(no)/bar_sigma - bar_sigma_a_ratio))))) 
                   -no)) # A checker : 1 pas dans la ppf
    

def gauss_of_log(s,a, no, bar_sigma):
    ''' Computes phi(alog(no+s)/bar_sigma) 
    - s (float): The variable with respect to which integration will be performed (used for the quadrature method)
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - no (positive int): The first record-breaking time
    - bar_sigma (positive float): The biggest variance term of the variance-covariance matrix
    --------------------------------------------------------------------------
    returns (int): The function evaluated in s 
    '''
    return norm.pdf(a*np.log(no+s)/bar_sigma)


def compute_go_x(k,a,no,bar_sigma):
    ''' Estimates the integral of gauss_of_log between k-1 and k and between 0 and infinity with quadrature techniques. 
    Then computes the ratio of those two quantities in order to obtain a probability.
    - k (float): Defines the bounds of the integral in the numerator
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - no (positive int): The first record-breaking time   
    --------------------------------------------------------------------------
    returns: (int): The estimated probability mass
    '''
    return quad(gauss_of_log,k-1,k,args=(a, no, bar_sigma))[0]/quad(gauss_of_log,0,np.inf,args=(a, no, bar_sigma))[0]


#===========================================================
# Simulate the Poisson Arrival Process
#===========================================================

def simulate_An(length, Ao=0, LAMBDA=1):
    ''' Simulate a Poisson Arrival Process of size <length>, starting point Ao and of rate LAMBDA
    - length (positive int): size of the Poisson process to simulate
    - Ao (int): The starting point of the Poisson process
    - LAMBDA (float): The Poisson process rate
    -----------------------------------------------------------------------------------------------    
    returns (1xlength array): A1, ..., An (and not Ao) 
    '''
    return  Ao + np.random.poisson(LAMBDA, size=length).cumsum() 


def simulate_S(length=500, So=0, LAMBDA=1, gamma=.5):
    ''' Compute a random walk of size length
    - length (positive int): size of the Poisson process to simulate
    - Ao (int): The starting point of the Random Walk process
    - LAMBDA (float): The Poisson process rate
    - gamma (float in (0,1)): parameter of the random Walk chosen such that gamma < E(A_1)
    -----------------------------------------------------------------------------------------------    
    returns (1xlength array): S1, ..., Sn (and not So) 
    '''
    An = simulate_An(length=length,Ao=-So,LAMBDA=LAMBDA) # Compute An
    return (gamma*np.arange(1,length+1) - An) # Check right shape



def sample_downcrossing(x):
    ''' Simulate a downcrossing segment from a random Walk S with a negative drift
    x (int): Starting point of the downcrossing segment
    -----------------------------------------------------------------------------------------------    
    returns (1xtau_minus array): S1, ..., S_{tau_minus}, with tau_minus The lowest n such that Sn<0  
    '''
    Sn = simulate_S(So=x, LAMBDA=1)
    tau_minus = np.argmax(Sn<0) 
    return Sn[:tau_minus+1].tolist()

def sample_upcrossing(x):
    ''' Simulate a upcrossing segment from a random Walk S with a non-negative drift
    - x (int): Starting point of the upcrossing segment
    -----------------------------------------------------------------------------------------------    
    returns (1xtau_plus array): S1, ..., S_{tau_plus}, with tau_plus The lowest n such that Sn>0 
    '''
    theta = compute_theta()
    Sn = simulate_S(So=x, LAMBDA=1+theta)

    S = Sn[:np.argmax(Sn>=0)+1] 
    u = np.random.uniform(high=1,low=0,size=1)[0]
    
    #print("cond", np.exp(-theta*(S[-1]-x)))
    if u<np.exp(-theta*(S[-1]-x)):
        return S.tolist()
    else:
        return 'degenerated'


def algorithm_S_NA(): 
    ''' Generate random walk values S1, ..., S_NA, with NA the random time such that 
    for all n>NA An>gamma*n i.e Sn<0 
    -----------------------------------------------------------------------
    returns (1xNA array): S1, ..., S_NA
    '''
    S = [0.]
    degenerated = False
    while degenerated!=True: # Alternate downcrossing and upcrossing segment until the next segment is degenerated
        down_crossing_segment = sample_downcrossing(S[-1])
        S = S + down_crossing_segment
        up_crossing_segment = sample_upcrossing(S[-1])
        
        if up_crossing_segment!='degenerated':
            S = S + up_crossing_segment
        else:
            degenerated=True
    return S


def sample_without_record(x,l):
    ''' Sample S_{N_A+1} to S_N conditionally on tau_plus being infinite.
    - x (negative int): Starting point of the segment
    - l (positive int): length of the segment to simulate (=N-N_A). 
    -----------------------------------------------------------------------
    returns (1xl array): S_{N_A+1}, ..., S_N
    '''
    S = simulate_S(length=l, So=0, LAMBDA=1) 
    while (np.max(S)>=0) or (sample_upcrossing(S[-1])!='degenerated'): # Triggers a useless warning
        S = simulate_S(length=l, So=x, LAMBDA=1)
    return S.tolist()


def algorithm_S_NA_to_N(S,l):
    ''' Useless function implemented to mimic the implementation of the paper, will be deleted later on
    Concatenates S_1,..., S_{N_A} and S_{N_A+1},..., S_N.
    S (1xNA array): The Random Walk of length NA
    l (positive int): length of the segment to simulate (=N-N_A). 
    -----------------------------------------------------------------------
    returns (1xN array): S_1, ..., S_N    
    '''
    if l>0:
        S = S + sample_without_record(S[-1],l)
    return S


#===========================================================
# Gaussian random fields related simulation
#===========================================================

def simul_Xk(t, size=1):
    ''' Simulates a multivariate centered brownian motion of step dt=1 starting at (0,...,0). 
    - cov (dxd ndarray): Variance-covariance matrix of the random fields, with d is the dimension of the multivariate process.
    - size (positive int): The length of the process to simulate
    ----------------------------------------------------------------------------
    returns: (sizexd ndarray): The multivariate Brownian motion of size <size> 
    '''
    # Generate the sequence of multivariate_gaussian of size "size"
    nb_points = len(t)
    dx = [t[0]] + [t[i+1]-t[i] for i in range(nb_points-1)] # range is now including the right bound
    
    r = multivariate_normal.rvs(cov=np.diag(dx), size=(size))
    
    if size>1:
        return np.cumsum(r, axis=-1)
    else:  
        return r


def cov_brownian_X(dist, sigma=1):
    ''' Simulate the variance-covariance matrix of the brownian motion '''
    nb_points = len(dist)
    
    cov = np.empty(shape=(nb_points, nb_points))
    for i in range(nb_points):
        for j in range(nb_points):
            cov[i][j] = sigma*np.sum(dist[0:min(i,j)+1])
            
    return cov


def modified_bernoulli(p=.5):
    ''' Draw a single draw from a Bernoulli that returns -1 or 1 instead of 0 or 1 with probability p
    - p (float in [0,1]) probabilty of the Bernoulli
    -----------------------------------------------------------------------
    returns (int): +1 with probabilty p and -1 with probability 1-p
    '''
    x = np.random.binomial(n=1,p=0.5)
    x = x if x>0 else -1
    return x

def conditioned_sampleX(a,n, cov, t): 
    ''' Sample X_k such that ||X_k||_inf > alogk
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - n (positive int): The record breaking time from which to sample X
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    -----------------------------------------------------------------------
    returns (1xd array): X_k   
    '''
    alogn = a*np.log(n)
    
    location_weights = 2*(1- np.array([norm.cdf(alogn, loc=0, scale=sig) for sig in np.sqrt(np.diag(cov))]))
    tv = multinomial(n=1, pvals=location_weights/location_weights.sum()) 
    tv = np.where(tv==1)[0][0] # We choose an index in 1,...,d according to the previous multinomial
    
    u = np.random.uniform(high=1,low=0,size=1)[0]
    j = modified_bernoulli()
    
    sigma_tv = np.sqrt(cov[tv,tv])

    X_tv = sigma_tv*j*norm.ppf(u+(1-u)*norm.cdf(alogn/sigma_tv))
    Y =  simul_Xk(t) # Why don't we take X_{K-1} to start the sequence ? Maybe because of the independence of the X_{n}s

    return Y - cov[:,tv]*Y[tv]/cov[tv,tv] + X_tv # Returns X_k


def sample_single_record(a,ni,n_i_plus_1, cov, t):  # Does not depend on delta.. (Only through a)
    ''' Sample X1, ..., X_{T_{n_i_plus_1}} according to the paper. (Seems to me that it rather sample X_ni,..., X_{n_i_plus_1})
    - ni (int): The ith record-breaking time
    - n_i_plus_1 (int): The i+1th record-breaking time
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    -----------------------------------------------------------------------
    returns (T_{n_i_plus_1}xd ndarray or str): X_1, ..., X_{T_{n_i_plus_1}} or "degenerated" if the sample is improper
    '''
    bar_sigma = np.sqrt(np.max(np.diag(cov))) 

    K = compute_K(a,ni, bar_sigma) # Compute the random time K 
    #print("K=", K)
    X = simul_Xk(t, size=K-1) # Compute X_1,...,X_{K-1}
    X_K  = conditioned_sampleX(a,n_i_plus_1+K,cov,t) # Compute X_K
    
    if K>2: # Deals with numpy array structures (can be improved)
        X = np.append(X.tolist(), [X_K.tolist()], axis=0) 
    elif K==2:
        X = np.stack([X,X_K])
    else:
        X = np.array([X_K])
    
    u = np.random.uniform(high=1,low=0,size=1)[0]

    # Compute the result of the first condition 
    cond1 = all(nl.norm(X[:-1], ord=np.inf,axis=1) <= a*np.log(n_i_plus_1+np.arange(1,K)))

    # Compute the result of the second condition 
    theoretical_proba = 2*(1- np.array([norm.cdf(a*np.log(n_i_plus_1+K), loc=0, scale=sig) for sig in np.sqrt(np.diag(cov))])).sum()
    empirical_proba = np.sum(abs(X_K)>a*np.log(n_i_plus_1+K))
    cond2 = u*compute_go_x(K,a,ni,bar_sigma) < theoretical_proba/empirical_proba 

    if cond1 and cond2: # If both conditions are respected return the corresponding sequence
        return X
    else:  # Else the generated sample is improper
        return 'degenerated'
    
    
def algorithm_X_NX(a, cov, t):
    ''' Sample X1, ... , X_NX 
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    -----------------------------------------------------------------------
    returns (N_Xxd ndarray): X_1, ..., X_NX 
    '''
    d = cov.shape[1]
    bar_sigma = np.sqrt(np.max(np.diag(cov)))
    no = compute_no(a, bar_sigma, d)
    eta = no
    X = simul_Xk(t, size=eta) 
    segment = True 
    
    while segment!= 'degenerated': # The while+condition loop from the paper is pretty ugly, have to be changed
        segment = sample_single_record(a,no,eta, cov,t)
        if segment!='degenerated':
            X = np.append(X.tolist(), segment.tolist(), axis=0)
            eta = len(X)
    return X

def sample_without_recordX(a,ni,l,t): # Typo in the paper ? 
    ''' Sample X_{N_X+1}, ..., X_N
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - n1 (int): The ith record-breaking time
    - l (positive int): the length of the sequence to generate, one has l=N-N_X
    -----------------------------------------------------------------------
    returns ((N-N_X)xd ndarray): X_{N_X+1}, ..., X_N 
    '''    
    X = simul_Xk(t, size=l) 
    
    # check X is not of dim 1
    while (nl.norm(X, ord=np.inf,axis=1) - a*np.log(ni+np.arange(1,l+1))>0).any(): # Typo for the infinite norm 
        X = simul_Xk(t, size=l) # Why don't we start from X_{ni-1}, does not break the dynamic ? 
    return X

def algorithm_X_NX_to_N(a,X,l, t):
    ''' Useless function implemented to mimic the paper. Simply concatenate X_1,...,X_NX with X_{NX+1}, ..., X_N
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - X (N_{X}xd ndarray): Sequence of X1, ..., X_{N_X}
    -----------------------------------------------------------------------
    returns (Nxd ndarray): X_1, ..., X_N     
    '''
    eta = X.shape[0]
    if l>0: 
        X = np.append(X.tolist(), sample_without_recordX(a,eta, l, t).reshape(-1,X.shape[1]).tolist(), axis=0)
        return X

    else:
        raise ValueError('Please enter a positive integer value for l')

#===========================================================
# Max-stable random fields simulation
#===========================================================

def algorithm_M(a, cov, t, gamma=.5, return_rnd_times=False):
    ''' Generates the max-stable random fields vector 
    - a (float in (0,1)): Parameter of the model obtained from compute_a(.)
    - cov (dxd ndarray): Variance-covariance matrix of the random fields
    - x (array-like): coordinates of interest
    - return_rnd_times (bool): If True returns the vector of max-stable random fields but also, N,N_X, N_A and Na. 
        Otherwise it only returns M,N
    -----------------------------------------------------------------------
    returns ((1xd array,int) tuple): M=(M(t1),..., M(t_d)),N or M,N,Na,N_X,N_A if return_rnd_times==True       
    '''
    S =  algorithm_S_NA() # Simulate S_1,..,S_NA
    A = gamma*np.arange(1,len(S)+1) - S # Retrieve A from S
    d = cov.shape[0]
    X = algorithm_X_NX(a, cov, t).reshape(-1,d) # Simulate X_1,...,X_NX
    
    Na, N_X, N_A = compute_Na(d, cov, a, A[0], X[0,:], gamma), len(X), len(A) # Compute the random times
    N = max(Na, N_X, N_A)
    print("N: ", N, "Na: ", Na, "N_A: ", N_A)
    if N>N_A: # Simulate S_{NA+1},.., S_N
        S =algorithm_S_NA_to_N(S,N-N_A) 
        A = gamma*np.arange(1,len(S)+1) - S
        
    if N>N_X:# Simulate X_{NX+1},.., X_N
        X = algorithm_X_NX_to_N(a,X,N-N_X,t)

    M_t_1_n = -np.log(np.stack([A]*X.shape[1]).T) + X # Compute -log(An) + X_n(t) for all n in 1,..., N
    
    if return_rnd_times:
        return np.max(M_t_1_n, axis=0), X, N, Na, N_X, N_A 
    else:
        return np.max(M_t_1_n, axis=0), X, N 