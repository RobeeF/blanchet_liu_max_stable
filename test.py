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

t = [1/3,1/3,1/3]
cov = cov_brownian_X(t)
bar_sigma = np.sqrt(max(np.diag(cov)))
a = 0.5#compute_a(cov, delta)
no = compute_no(a,bar_sigma,d)

#===========================================================
# Moments analysis
#===========================================================
### Check NA
S = algorithm_S_NA()
S_NA = sample_without_record(S[-1],1000)

n_gamma = gamma*np.arange(1,len(S+S_NA)+1)
A_NA = n_gamma - (S+S_NA)

pd.Series(A_NA).plot() # Check that Sn for all n in [NA,N] is negative OK
pd.Series(n_gamma).plot()
N_A = len(S)
print(N_A)

### Check N_a  
a = compute_a(cov, delta) # a=0.5

a=0.5
A1 = simulate_An(1)[0]
X1 = simul_Xk(t, size=1) # Generate  
print(X1, A1)

N_a = compute_Na(d,cov,a,A1,X1)
print(N_a)
n = np.arange(max(N_a-100,0),N_a+100)

pd.Series(n*gamma).plot()
pd.Series(A1*np.power(n,a)*np.exp(nl.norm(X1, ord=np.inf))).plot()
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