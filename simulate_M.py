# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:32:59 2019

@author: robin
"""


from scipy.stats import norm, multivariate_normal
from numpy.random import multinomial


#===========================================================
# Gaussian random fields simulation
#===========================================================

def simul_Xk(cov, size=1): # To change to a Brownian motion
    return multivariate_normal.rvs(cov=cov, size=size)


#===========================================================
# Compute useful constants (not finished)
#===========================================================
    
def compute_theta(Ao=0, LAMBDA=1, gamma=.5): # Theta equal to the lowest linspace value...
    NUM = 5
    theta = np.linspace(start=0.000001,stop=1, num=NUM)
    NB_SAMPLE = 100000
    A1 = Ao + np.random.poisson(LAMBDA, size=(NB_SAMPLE,NUM))
    S1 = -A1 + gamma 
    dist = np.abs(np.exp((theta*S1)).mean(axis=0)-1)
    #return  theta[np.where(dist==min(dist))[0][0]]
    return 0.6


def compute_Na(): 
    # To fill
    return 1

def compute_no(a, cov): # Not working currently
    bar_sigma = np.sqrt(np.max(np.diag(cov)))
    u = np.random.uniform(high=1,low=0,size=1)[0]
    #u*(1-norm.pdf(a*np.log()) - bar_sigma/a)
    #a*np.log(no)
    return 1

def compute_K(a, no, bar_sigma):
    u = np.random.uniform(high=1,low=0,size=1)[0]

    return int(np.ceil(
            np.exp((bar_sigma/a)**2 +
                   (bar_sigma/a)*(1 - norm.ppf(u*(1-norm.pdf(a*np.log(no)/bar_sigma - bar_sigma/a))))) 
                   -no))
    
def compute_go_x():
    # To fill
    return 1

def compute_a():
    # To fill
    return .5

#===========================================================
# Simulate the Poisson Arrival Process
#===========================================================

def simulate_An(length, Ao=0, LAMBDA=1):
    ''' Returns A1, ..., An and not Ao '''
    return  Ao + np.random.poisson(LAMBDA, size=length).cumsum() 

def simulate_S(length=500, So=0, LAMBDA=1):
    ''' Returns S1, ..., An and not So '''
    gamma=.5
    # Compute An
    An = simulate_An(length=length,Ao=-So,LAMBDA=LAMBDA)
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
    X_tv = sigma_tv*j*norm.ppf(u+(1-u)*norm.cdf(alogn/sigma_tv, loc=0))
    Y =  simul_Xk(cov) 
    return Y - cov[:,tv]*Y[tv] + X_tv # X_tv is X_tv*cov[:,tv] in (10)


def sample_single_record(a,no,n1, cov): 
    bar_sigma = np.sqrt(np.max(np.diag(cov))) # Max over the locations or over T for is ti..?

    K = compute_K(a,no, bar_sigma)
    K = min([K,7000]) # Quick fix, to remove

    X = simul_Xk(cov=cov, size=K) 
    X_K  = conditioned_sampleX(a,n1+K,cov)
    X = np.append(X.tolist(), [X_K.tolist()], axis=0) # Not optimal to go through list
    
    u = np.random.uniform(high=1,low=0,size=1)[0]

    cond1 = all([np.max(np.abs(X[k,:]))<=a*np.log(n1+k+1) for k in range(0,K-1)]) 
        
    theoretical_proba = 2*(1- np.array([norm.cdf(a*np.log(n1+K), loc=0, scale=sig) for sig in np.sqrt(np.diag(cov))])).sum()
    empirical_proba = np.sum(abs(X_K)>a*np.log(n1+K))
    
    cond2 = u < theoretical_proba/empirical_proba # Not sure for cond2 that just inversing the ratio is ok
    if cond1 and cond2:
        return X
    else: 
        return 'degenerated'

def sample_without_recordX(a, n1,l, cov): # Dimension problem ?
    X = simul_Xk(cov=cov, size=l) 
    while all(np.max(X - a*np.log(n1+np.arange(1,l+1)).reshape(-1,1), axis=1)>0): # Dimension problem max for several dimension..
        X = simul_Xk(cov=cov, size=l) 
    return X


def algorithm_X_NX(a, cov):
    no = compute_no(a,cov)
    eta = no
    X = simul_Xk(cov=cov, size=eta) 
    segment = True # Sale, Boucle do while ?
    
    while segment!= 'degenerated': # Boucle while+condition a pas trop de sens
        segment = sample_single_record(a,no,eta, cov)
        if segment!='degenerated':
            X = np.append(X.tolist(), segment.tolist(), axis=0)
            X += segment
            eta = len(X) 
    return X


def algorithm_X_NX_to_N(a,X,l, cov):
    eta = X.shape[0]
    if l>0: 
        X = np.append(X.tolist(), sample_without_recordX(a,eta, l, cov).reshape(-1,X.shape[1]).tolist(), axis=0)
        return X

    else:
        raise ValueError('Please enter a positive integer value for l')

def algorithm_M(a, cov, gamma=.5):
    S =  algorithm_S_NA() 
    A = gamma*np.arange(1,len(S)+1) - S
    d = cov.shape[0]
    X = algorithm_X_NX(a, cov).reshape(-1,d)
    
    Na, N_X, N_A = compute_Na(), len(X), len(A)
    N = max(Na, N_X, N_A)

    if N>N_A:
        S =algorithm_S_NA_to_N(S,N-N_A) 
        A = gamma*np.arange(1,len(S)+1) - S
    if N>N_X:
        X = algorithm_X_NX_to_N(a,X,N-N_X, cov)
    
    M_t_1_n = -np.log(np.stack([A]*X.shape[1]).T) + X 
    return np.max(M_t_1_n, axis=0), X, N 