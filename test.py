# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 11:52:05 2019

@author: Robin Fuchs
"""

import os

os.chdir('C:/Users/robin/Documents/GitHub/blanchet_liu_max_stable')


from estimator import *
import numpy as np
from simulate_M import *
import pandas as pd
import numpy.linalg as nl
import matplotlib.pyplot as plt
from numpy.linalg import det, inv
from scipy.linalg import sqrtm
import pickle

# Standard setup
d=3
delta = 0.05
gamma =0.5

t = [1/3,2/3,3/3]
dist = [1/3,1/3,1/3]
cov = cov_brownian_X(dist) # Probleme with the new t
bar_sigma = np.sqrt(max(np.diag(cov)))
a = 0.5#compute_a(cov, delta)
no = compute_no(a,bar_sigma,d)

#===========================================================
# Moments analysis
#===========================================================
### Check NA
S = algorithm_S_NA()
S_NA = sample_without_record(S[-1],50)

n_gamma = gamma*np.arange(1,len(S+S_NA)+1)
A_NA = n_gamma - (S+S_NA)

print(N_A)
plt.figure(figsize=(7,3))
plt.plot(A_NA, lw=2)
plt.plot(n_gamma, lw=2)
plt.title("For n>NA, Sn= gamma*n-An<0")
plt.xlabel("n")
plt.ylabel("Value")
plt.legend(["An", "n*gamma"])
plt.tight_layout()

### Check N_a  
a = compute_a(cov, delta) # a=0.5

A1 = simulate_An(1)[0]
X1 = simul_Xk(t, size=1) # Generate  

N_a = compute_Na(d,cov,a,A1,X1)
n = np.arange(max(N_a-100,0),N_a+100)

print(N_a)
plt.figure(figsize=(7,3))
plt.plot(A1*np.power(n,a)*np.exp(nl.norm(X1, ord=np.inf)), lw=2)
plt.plot(n*gamma, lw=2)
plt.title("For n>Na, n*gamma > A1*n^a*exp(||X1||inf)...")
plt.xlabel("n")
plt.ylabel("Value")
plt.legend(["A1*n^a*exp(||X1||)", "n*gamma"])
plt.tight_layout()

# gamma*n> A_1*(n^a)*exp(||X_1||inf) for n>Na : OK

### Check N_X
n1 = 2000
X_NX_to_N = sample_without_recordX(a,n1,400,t)
norm_X_NX_to_N = nl.norm(X_NX_to_N, ord=np.inf,axis=1) 
a_log_n = a*np.log(n1+np.arange(1,500))

pd.Series(norm_X_NX_to_N).plot()
pd.Series(a_log_n).plot()
# Ok


#==================================================================
# Check that single_record outputs degenerated 1-delta % of the time
#==================================================================
iters = 1000
non_degenerate=0
for i in range(1,iters):
    print(i)
    if type(sample_single_record(0.5,no,no, cov,t)) != str:
        non_degenerate+=1
print(100*(1- non_degenerate/iters), "% of degenerated")

#===========================================================
# Printing L distribution
#===========================================================

L_sample = []
for i in range(10000):
    L_sample.append(simulate_L()) 
    
# Take the distribution without the tail for a better display
distrib_L_without_tail = pd.Series(L_sample).value_counts()[:50]
(pd.Series(distrib_L_without_tail)/np.sum(distrib_L_without_tail)).plot()


#============================================================
# Check the Cramer's root computation
#============================================================
A1 = np.random.poisson(1, size=(100000)) 
S1 = 1/2 - A1  
theta = compute_theta()
np.mean(np.exp(theta*S1))

# OK


#====================================================================================
# Verify the length of the downcrossing segment compared to the whole segment length
#====================================================================================

downc = np.array(sample_downcrossing(0))
# Implemented a warning

#====================================================================================
# Same thing for the upcrossing segment
#====================================================================================

simulate_An(5, Ao=0, LAMBDA=100)
while  str(np.array(sample_upcrossing(-1))) == 'degenerated':
    pass
upcross = np.array(sample_upcrossing(-1))
Sn = simulate_S(So=-10, LAMBDA=1+theta)

# Problem fixed with an error raised

#=============================================
# Test with sample without record
#=============================================
sample_without_record(-3,10)

#=============================================
# Test with sample without record
#=============================================
up_down_seq = algorithm_S_NA()
up_down_seq
pd.Series(up_down_seq).plot()

#=============================================
# Full S algo
#=============================================
up_down_seq = algorithm_S_NA()
algorithm_S_NA_to_N(up_down_seq,10)

#===========================================================
# How does a vary with delta?
#===========================================================
# Non monotonic:
a_delta_values = []
linspace = np.linspace(0.000001,0.9999,1000)
for de in linspace:
    a_delta_values.append(compute_a(cov,delta))
pd.Series(a_delta_values).plot() # Looks like a white noise of mean 0.6--

#===========================================================
# How does no vary with a?
#===========================================================
# Non monotonic:
no_a_values = []
linspace = np.linspace(0.45,0.8,1000)
for a in linspace:
    no_a_values.append(compute_no(a,bar_sigma,d))

plt.plot(linspace, no_a_values) # No decreases with a

#===========================================================
# Compute kde estimation
#===========================================================
b = 200
a = 0.5 # Take delta equal to 0.05 while it has no influence
A = cov/np.linalg.det(cov)
h_b = b**(-1/(2*d+1))

x_minus_M_L = [] # Will store the sequence of M^(i)-x for all i in 1,...,L 
x = np.full(d,0)

for i in range(b):
        print(i,' eme simulation')
        M,X,N = algorithm_M(a, cov, t)
        x_minus_M = x - M
        x_minus_M_L.append(x_minus_M)

x_minus_M_L = np.stack(x_minus_M_L)


f = np.sum(norm.pdf(np.dot(sqrtm(inv(A)),(x_minus_M_L).T)/h_b))/(b*(h_b)**d)

#============================================================
# Plot M distribution 
#============================================================

M_list = []
for i in range(100): 
    print(i)
    M,X,N = algorithm_M(a, cov,t, gamma=.5)
    M_list.append(M)
    
np_M_list = np.stack(M_list)
np_M_list.shape

pd.Series(np_M_list[:,0]).plot('kde')
pd.Series(np_M_list[:,1]).plot('kde')
pd.Series(np_M_list[:,2]).plot('kde')

print(np_M_list[:,0].mean())
print(np_M_list[:,1].mean())
print(np_M_list[:,2].mean())

with open('M_1580_simul_070319.pkl', 'wb') as fichier:
    mon_pickler = pickle.Pickler(fichier)
    mon_pickler.dump(M_list)

# Get the pickled objects 
with open('M_1580_simul_070319.pkl', 'rb') as fichier:
        mon_depickler = pickle.Unpickler(fichier)
        output_depickled = mon_depickler.load()
    
#============================================================
# Schema explicatif 
#============================================================        

fake_Sn = [0,1,3,2,-2, -1, -3, -2, -3, -2, 1, 2, 4, 1, -5, -2, -4, -3, -2, -2, -1,-2,-3,-3,-2]   
pd.Series(fake_Sn).plot() 
    
