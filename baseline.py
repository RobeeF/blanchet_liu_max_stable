# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:46:29 2019

@author: robin
"""

import os 
os.chdir('C:/Users/robin/Documents/GitHub/blanchet_liu_max_stable')

from simulate_M import simul_Xk, simulate_An
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import kv, gamma # Check third kind for kv


#============================================================
# Estimating P(M(t1)<x1, M(t2)<x2,...,M(td)<xd) through (2) 
#============================================================

d = 3
t = [(1/3)*i for i in range(1,d+1)]
x = np.full(d,6)
X = simul_Xk(t, size=10000)

F_baseline = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
print(F_baseline) 

#============================================================
# Utilities
#============================================================

def rho(h,v,c):
    ''' Mattern kernel, to finish'''
    corr_matrix = ((2**(1-v))/gamma(v))*((h/c)**(v))*kv(v,h/c)
    np.fill_diagonal(corr_matrix,1)
    return corr_matrix


def cdf(M,x):
    ''' Compute the empirical cdf of the data'''
    return sum((M<=x).all(axis=1))/len(M) # Check axis=1

def dist_matrix(t):
    ''' Compute the euclidian distance between the locations '''
    d = len(t)
    epsilon = 10**(-7) # To avoid explosion when the Bessel function is called
    mtx = np.empty((d,d))
    for i in range(d):
        for j in range(d):
            mtx[i][j] = np.linalg.norm(t[i]-t[j]) + epsilon
    return mtx

#============================================================
# Empirical data importation and treatment
#============================================================
    
rainfall = pd.read_csv("rainfall.csv")
rainfall = rainfall.iloc[:,1:].values
d = rainfall.shape[1]

mu = rainfall.mean(axis=0)
M = (rainfall - mu) # Centered
M/=np.sqrt(np.var(M,axis=0)) # Standardisation

x = np.full(d,6)
emp_cdf = cdf(M,x)

coord = pd.read_csv("coord.csv").values[:,1:3]
dist_list = [np.linalg.norm(coord[i+1]-coord[i]) for i in range(d-1)] # Distance from loc1 to loc2, loc2 to loc3 etc...
dist_list = [np.linalg.norm(coord[0])] + dist_list

dist_mtx = dist_matrix(coord)


#============================================================
# Matern-Whithney kernel: integrate the square distance
#============================================================



d = M.shape[1]

min_M_emp = np.min(M,axis=0)
max_M_emp = np.max(M,axis=0)

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.random.uniform(min_M_emp+1,max_M_emp+5, size=(200,d))
params_grid = np.mgrid[5:12.1:0.6, 7:13.1:0.6].reshape(2,-1).T

x = x_grid[8]
X = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx, 20,10))
cdf(M,x)
np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))


th_cdf_curves = {}
cdf_dist = {}


for params in params_grid:
    print(params)
    cdf_dist[str(params)] = []
    X = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx, params[0],params[1]))

    for x in x_grid: 
        # Compute the theoretical and empirical cdfs
        emp_cdf = cdf(M,x)
        th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))

        # Store the squared euclidian distance for all x and the current params
        current_dist = (emp_cdf-th_cdf)**2
        cdf_dist[str(params)].append(current_dist)
        
# Compute the mean distance for each parameters couple
for params in params_grid:
    cdf_dist[str(params)] = np.mean(cdf_dist[str(params)])

# Define a threshold to keep only the 5% of the lowest distance associated parameters
accept_threshold = np.quantile(list(cdf_dist.values()), 0.10)

params_distrib = []
for params in params_grid:
    if cdf_dist[str(params)]<accept_threshold:
        params_distrib.append(params)
    
# Plot the best couples of parameters
params_distrib = np.stack(params_distrib)
v,c = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(v, c, bins=(3, 3), cmap=plt.cm.BuPu)

# Have a look at the marginal densities
pd.Series(v).plot("kde")
pd.Series(c).plot("kde")


#============================================================
# Matern-Whithney kernel: Try with 2 locations
#============================================================
d = 2

min_M_emp = np.min(M,axis=0)[:2]
max_M_emp = np.max(M,axis=0)[:2]

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.mgrid[min_M_emp[0]:max_M_emp[0]+0.1:0.2, min_M_emp[1]:max_M_emp[1]+0.1:0.2].reshape(2,-1).T
params_grid = np.mgrid[19.5:22.5:0.4, 160:260.1:2].reshape(2,-1).T
params_grid.shape

th_cdf_curves = {}
cdf_dist = {}


for params in params_grid:
    print(params)
    cdf_dist[str(params)] = []
    X = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx, params[0],params[1])[:2,:2])

    for x in x_grid: 
        # Compute the theoretical and empirical cdfs
        emp_cdf = cdf(M[:,:2],x)
        th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))

        # Store the squared euclidian distance for all x and the current params
        current_dist = (emp_cdf-th_cdf)**2
        cdf_dist[str(params)].append(current_dist)
        
# Compute the mean distance for each parameters couple
for params in params_grid:
    cdf_dist[str(params)] = np.mean(cdf_dist[str(params)])

# Define a threshold to keep only the 5% of the lowest distance associated parameters
accept_threshold = np.quantile(list(cdf_dist.values()), 0.15)

params_distrib = []
for params in params_grid:
    if cdf_dist[str(params)]<accept_threshold:
        params_distrib.append(params)
    
# Plot the best couples of parameters    
params_distrib = np.stack(params_distrib)
v,c = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(v, c, bins=(4, 4), cmap=plt.cm.BuPu)

# Parameters Marginal densities
pd.Series(v).plot("kde")
pd.Series(c).plot("kde")


#====================================================================================================
# Matern-Whithney kernel: Two by two locations densities comparison (do not consider for the moment)
#====================================================================================================
d = 2

min_M_emp = np.min(M,axis=0)
max_M_emp = np.max(M,axis=0)

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.mgrid[min_M_emp[0]:max_M_emp[0]+0.1:0.2, min_M_emp[1]:max_M_emp[1]+0.1:0.2].reshape(2,-1).T

params = [7.8,10.2]

th_cdf_density = []
emp_cdf_density = []


X = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx[:2,:2], params[0],params[1]))

for x in x_grid: 
    # Compute the theoretical and empirical cdfs
    emp_cdf = cdf(M[:,:2],x)
    emp_cdf_density.append(emp_cdf)
    
    #print(emp_cdf)
    th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
    th_cdf_density.append(th_cdf)
    #print(th_cdf)
    
        
x_coords,y_coords = x_grid[:,0], x_grid[:,1]
plt.hist2d(x_coords, y_coords, weights=emp_cdf_density, cmap=plt.cm.BuPu)

#====================================================================================================
# Matern-Whithney kernel: Testing the approach
#====================================================================================================
def simul_M(seq_len, cov):
    ''' Approximately simulate M according to its definition (by ) '''
    d = cov.shape[0]
    An = simulate_An(seq_len)
    Xn = np.random.multivariate_normal(size=seq_len, mean=np.full(d,0), cov=cov)
    return np.max(-np.log(np.stack([An]*Xn.shape[1]).T) + Xn, axis=0) 

x = x_grid[4]

M_simu = []
for i in range(5000):
    print(i)
    M_simu.append(simul_M(100000, rho(dist_mtx, 1,10)))

M_simu2_np = np.stack(M_simu)
M_simu_np.shape

M_simu_np = np.append(M_simu_np, M_simu2_np, axis=0)
M_simu_np = np.stack(M_simu) 
pd.DataFrame(M_simu_np).to_csv("simu_M_25000.csv")

M_simu_np.shape

M_df[M_df>np.full(d,90)].shape
x = np.full(d, 7)
cdf(M_simu,x)

X = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx, 1,10))
np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
