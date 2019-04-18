# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:46:29 2019

@author: robin
"""

import os 
os.chdir('C:/Users/robin/Documents/GitHub/blanchet_liu_max_stable')

from simulate_M import simul_Xk, cov_brownian_X
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import kv, gamma # Check third kind for kv
from sklearn.linear_model import LinearRegression


#============================================================
# Estimating P(M(t1)<x1, M(t2)<x2,...,M(td)<xd) through (2) 
#============================================================

d = 3
t = [(1/3)*i for i in range(1,d+1)]
x = np.full(d,6)
X = simul_Xk(t, size=10000)

F_baseline = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
print(F_baseline) # Grosse variabilité

#============================================================
# Utilities
#============================================================

def rho(h,v,c):
    ''' Mattern kernel, to finish'''
    corr_matrix = ((2**(1-v))/gamma(v))*((h/c)**(v))*kv(v,h/c)
    np.fill_diagonal(corr_matrix,1)
    return corr_matrix


def cdf(X,x):
    ''' Compute the empirical cdf of the data'''
    return sum((X<=x).all(axis=1))/len(X) # Check axis=1

def dist_matrix(t):
    d = len(t)
    epsilon = 10**(-7) # To avoid explosion
    mtx = np.empty((d,d))
    for i in range(d):
        for j in range(d):
            mtx[i][j] = np.linalg.norm(t[i]-t[j]) + epsilon
    return mtx

x_grid = np.random.uniform(min_X_emp,max_X_emp, size=(20,d))
x = x_grid[0]

X_synth = np.random.multivariate_normal(size=20000, mean=np.full(d,0), cov=rho(dist_mtx, 1, 10)[:5,:5])
np.mean(np.max(np.exp(X_synth-x), axis=1))

np.mean(np.var(np.exp(X_synth-x), axis=1))
np.exp(-(np.mean(np.max(np.exp(X_synth-x), axis=1))))
rho(dist_mtx, 4,4)
#============================================================
# Empirical data importation and treatment
#============================================================
    
rainfall = pd.read_csv("rainfall.csv")
rainfall = rainfall.iloc[:,1:].values
d = rainfall.shape[1]

mu = rainfall.mean(axis=0)
M = (rainfall - mu)
M/=np.sqrt(np.var(M,axis=0))

x = np.full(d,6)
emp_cdf = cdf(M,x)

coord = pd.read_csv("coord.csv").values[:,1:3]
dist_list = [np.linalg.norm(coord[i+1]-coord[i]) for i in range(d-1)] # Distance from loc1 to loc2, loc2 to loc3 etc...
dist_list = [np.linalg.norm(coord[0])] + dist_list

dist_mtx = dist_matrix(coord)

#============================================================
# Specification test for Brownian motion
#============================================================

x_grid = np.arange(0,70,3)
sigma_grid = np.linspace(0.0001,0.003375,5)

th_cdf_curves = {}

for sigma in sigma_grid:
    print(sigma)
    th_cdf_curves[sigma] = []
    emp_cdf_curves = []
    X = np.random.multivariate_normal(size=100000, mean=np.full(d,0), cov=cov_brownian_X(dist_list, sigma=sigma))

    for x_value in x_grid: 
        x = np.full(79,x_value)

        emp_cdf = cdf(M,x)
        emp_cdf_curves.append(emp_cdf)

        # Compute the theoretical and empirical cdfs
        th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
        th_cdf_curves[sigma].append(th_cdf)


for sigma in sigma_grid:
    plt.plot(x_grid, th_cdf_curves[sigma])
plt.plot(x_grid, emp_cdf_curves)

plt.show()

# Very bad specification...


#============================================================
# Sample generation and test performing for Brownian motion
#============================================================

cov_th = cov_brownian_X(dist_list, sigma=1) # 
d = len(dist_list)

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.random.uniform(0,7, size=(10,d))
# Better but unusable solution
#x_grid = np.mgrid[[slice(i,j,2) for i,j in [(-2,2.1)]*79]] 


cdf_dist = {}
for sigma in np.linspace(0.3,0.5,5): 
    cdf_dist[sigma] = 0
    for x in x_grid: 
        print("empirical cdf: ",emp_cdf)
        emp_cdf = cdf(M,x)
        # Compute the theoretical and empirical cdfs
        X = np.random.multivariate_normal(size=100000, mean=np.full(d,0), cov=cov_brownian_X(dist_list, sigma=sigma))
        th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
        print("Theoretical cdf: ", th_cdf)
        # Store the sup of the distance for all x and the current sigma, as in a Kolmogorov-Smirnov test (for all sigmas)
        current_dist = abs(emp_cdf-th_cdf)
        cdf_dist[sigma] = current_dist if current_dist>cdf_dist[sigma] else cdf_dist[sigma] 
 
#================================================================================
# Least squares between empirical and theoretical var-cov-coeffs for Brownian M.
#================================================================================

cov_emp_coeffs = np.cov(M.T).reshape(-1,1).flatten()
cov_b_coeffs = cov_brownian_X(dist_list).reshape(-1,1)
cov_emp_coeffs.shape
cov_b_coeffs.shape

lr = LinearRegression(fit_intercept=False)
lr.fit(cov_b_coeffs,cov_emp_coeffs)
best_sigma = lr.coef_[0]
coeffs_preds = lr.predict(cov_b_coeffs)

plt.scatter(cov_b_coeffs.flatten(), cov_emp_coeffs)

# Testing the optimal sigma value
x_grid = np.arange(0,70,3)

th_cdf_curves = []
emp_cdf_curves = []

X = np.random.multivariate_normal(size=100000, mean=np.full(d,0), cov=cov_brownian_X(dist_list, sigma=best_sigma))

for x_value in x_grid: 
    x = np.full(79,x_value)

    emp_cdf = cdf(M,x)
    emp_cdf_curves.append(emp_cdf)

    # Compute the theoretical and empirical cdfs
    th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
    th_cdf_curves.append(th_cdf)


plt.plot(x_grid, th_cdf_curves)
plt.plot(x_grid, emp_cdf_curves)

plt.show()

#============================================================
# Specification test for Matern-Whithney kernel
#============================================================

x_grid = np.arange(0,70,3)
sigma_grid = np.linspace(0.0001,0.003375,5)

params_grid = [[10,10], [15,10],[12,20],[4,10]]


th_cdf_curves = {}

for params in params_grid:
    print(params)
    th_cdf_curves[str(params)] = []
    emp_cdf_curves = []
    X = np.random.multivariate_normal(size=100000, mean=np.full(d,0), cov=rho(dist_mtx, params[0], params[1]))

    for x_value in x_grid: 
        x = np.full(79,x_value)

        emp_cdf = cdf(M,x)
        emp_cdf_curves.append(emp_cdf)

        # Compute the theoretical and empirical cdfs
        th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
        th_cdf_curves[str(params)].append(th_cdf)


for params in params_grid:
    plt.plot(x_grid, th_cdf_curves[str(params)])
plt.plot(x_grid, emp_cdf_curves)

plt.show()

# Test of the gaussian specification
th_cdf_curves = []
X = np.random.multivariate_normal(size=100000, mean=np.full(d,0), cov= np.cov(M.T))

for x_value in x_grid: 
    th_cdf = cdf(X,x)
    th_cdf_curves.append(th_cdf)

plt.plot(x_grid, th_cdf_curves)
plt.plot(x_grid, emp_cdf_curves)


#============================================================
# ABC Matern-Whithney kernel: integrate the square distance
#============================================================
d = len(dist_list)

min_M_emp = np.min(M,axis=0)
max_M_emp = np.max(M,axis=0)

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.random.uniform(min_M_emp+2,max_M_emp+5, size=(20,d))
params_grid = np.mgrid[1:8.1:0.2, 3:10.1:0.2].reshape(2,-1).T

th_cdf_curves = {}
cdf_dist = {}


for params in params_grid:
    print(params)
    cdf_dist[str(params)] = []
    X = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx, params[0],params[1]))

    for x in x_grid: 
        # Compute the theoretical and empirical cdfs
        emp_cdf = cdf(M,x)
        #print(emp_cdf)
        th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))
        #print(th_cdf)

        # Store the squared euclidian distance for all x and the current params
        current_dist = (emp_cdf-th_cdf)**2
        cdf_dist[str(params)].append(current_dist)
        #print("------------------------------------------------------------------")
        
# Compute the mean distance for each parameters couple
for params in params_grid:
    cdf_dist[str(params)] = np.mean(cdf_dist[str(params)])

# Define a threshold to keep only the 5% of the lowest distance associated parameters
accept_threshold = np.quantile(list(cdf_dist.values()), 0.05)

params_distrib = []
for params in params_grid:
    if cdf_dist[str(params)]<accept_threshold:
        params_distrib.append(params)
        
params_distrib = np.stack(params_distrib)
x,y = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(x, y, bins=(4, 4), cmap=plt.cm.BuPu)

pd.Series(x).plot("kde")
pd.Series(y).plot("kde")

#==================================================================
# Can re-Find the true parameters for ABC Matern-Whithney kernel ? How to simulate M from X ?
#==================================================================
#d = len(dist_list)
d = 79

cov=rho(dist_mtx, 4, 5)[:3,:3]

x = x_grid[3]
X_synth = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx, 3, 4))
cdf(X_synth,x)
np.exp(-(np.mean(np.max(np.exp(X_synth-x), axis=1))))

min_M_emp = np.min(M,axis=0)
max_M_emp = np.max(M,axis=0)

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.random.uniform(min_M_emp+1,max_M_emp, size=(20,d))
params_grid = np.mgrid[2:4.1:0.1, 3:6.1:0.1].reshape(2,-1).T
#params_grid = np.mgrid[1:6.1:0.3, 3:8.1:0.3].reshape(2,-1).T

th_cdf_curves = {}
cdf_dist = {}

for params in params_grid:
    print(params)
    cdf_dist[str(params)] = []
    X = np.random.multivariate_normal(size=30047, mean=np.full(d,0), cov=rho(dist_mtx, 3,4)[:5,:5])

    for x in x_grid: 
        # Compute the theoretical and empirical cdfs
        emp_cdf = cdf(X_synth,x)
        th_cdf = np.exp(-(np.mean(np.max(np.exp(X-x), axis=1))))

        # Store the squared euclidian distance for all x and the current params
        current_dist = (emp_cdf-th_cdf)**2
        cdf_dist[str(params)].append(current_dist)
        
# Compute the mean distance for each parameters couple
for params in params_grid:
    cdf_dist[str(params)] = np.mean(cdf_dist[str(params)])

# Define a threshold to keep only the 5% of the lowest distance associated parameters
accept_threshold = np.quantile(list(cdf_dist.values()), 0.03)

params_distrib = []
for params in params_grid:
    if cdf_dist[str(params)]<accept_threshold:
        params_distrib.append(params)
        
params_distrib = np.stack(params_distrib)
x,y = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(x, y, bins=(5, 5), cmap=plt.cm.BuPu)
