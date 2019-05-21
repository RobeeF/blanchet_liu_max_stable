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
import copy
import re
from numba import jit

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
    ''' Mattern kernel '''
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

def simul_M(seq_len, cov):
    ''' Approximately simulate M according to its definition (by ) 
    seq_len (int): Length of the X and A sequences used to approximate the infinite sequence '''
    d = cov.shape[0]
    An = simulate_An(seq_len)
    Xn = np.random.multivariate_normal(size=seq_len, mean=np.full(d,0), cov=cov)
    return np.max(-np.log(np.stack([An]*Xn.shape[1]).T) + Xn, axis=0) 


@jit(parallel=True)
def estim_params(M, dist_mtx, x_grid, params_grid, threshold, full_output=False, verbose=True):
    ''' Given an estimate of the parameters that are the more likely to have generated M '''
    d = M.shape[1]
    #X = np.random.multivariate_normal(size=50000, mean=np.full(d,0), cov=rho(dist_mtx, 20,10))
        
    cdf_dist = {}
    
    for params in params_grid:
        if verbose:
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
    accept_threshold = np.quantile(list(cdf_dist.values()), threshold)
    
    params_distrib = []
    for params in params_grid:
        if cdf_dist[str(params)]<accept_threshold:
            params_distrib.append(params)
    params_distrib = np.stack(params_distrib)
    
    if full_output:
        return cdf_dist 
    else:    
        return params_distrib


def find_inf_sup_search(dico):
    ''' Return the keys associated with the two smallest values of the dict dico for the second parameter of the grid'''
    dict_copy = copy.deepcopy(dico)
    min1 = min(dict_copy, key=dict_copy.get) # Extract the param couple associated with the smallest distance in str format
    min1_f = float(re.findall("([0-9]+.[0-9]+)\]$", min1)[0])
    del dict_copy[min1]
    
    min2 = min(dict_copy, key=dict_copy.get) # Extract the param couple associated with the 2nd smallest distance in str format
    min2_f = float(re.findall("([0-9]+.[0-9]+)\]$", min2)[0])

    return min(min1_f,min2_f), max(min1_f,min2_f) # Return in the right order


def dyadic_param_search(M, d, dist_mtx,nb_points = 800,SUP=300.1, INF=1, epsilon=1, verbose=False):
    
    min_M_emp = np.min(M,axis=0)[d-2:d]
    max_M_emp = np.max(M,axis=0)[d-2:d]
    
    n_iter = 100
    c_fixed = 5
    nb_values_tested = 20
    
    interv_len = SUP-INF
    
    
    # Initialisation
    x_grid = np.random.uniform(min_M_emp[0]+1,max_M_emp[1]+2, size=(nb_points,2))
    params_grid = np.mgrid[c_fixed:c_fixed+0.1:0.4, INF:SUP:interv_len/nb_values_tested].reshape(2,-1).T
    
    while((SUP-INF>epsilon) and n_iter>0):
        if verbose:
            print("Inf=",INF, "SUP=", SUP)
            
        params_distrib = estim_params(M[:,d-2:d], dist_mtx[d-2:d,d-2:d], x_grid, params_grid, full_output=True, threshold=1, verbose=False)
        INF, SUP = find_inf_sup_search(params_distrib) # Get the two best candidates and search in this region
        interv_len = SUP-INF # Legnth of the interval in which we are searching
        
        # Simulate the grid over which F and hat_F will be evaluated
        x_grid = np.random.uniform(min_M_emp[0]+1,max_M_emp[1]+2, size=(nb_points,2))
        params_grid = np.mgrid[c_fixed:c_fixed+0.1:0.4, INF:SUP:interv_len/nb_values_tested].reshape(2,-1).T
        
        nb_values_tested= max(3, nb_values_tested-7) # Test a lot of position at the beginning and fewer at the end
        n_iter-=1
        

    return estim_params(M[:,d-2:d], dist_mtx[d-2:d,d-2:d], x_grid, params_grid, threshold=0.00001, verbose=False)[0]

#============================================================
# Empirical data importation and treatment
#============================================================
    
rainfall = pd.read_csv("rainfall.csv")
rainfall = rainfall.iloc[:,1:].values
d = rainfall.shape[1]

rainfall.shape

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
# Matern-Whithney kernel: Whole 79 locations
#============================================================

d = M.shape[1]

min_M_emp = np.min(M,axis=0)
max_M_emp = np.max(M,axis=0)

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.random.uniform(min_M_emp+1,max_M_emp+5, size=(200,d))
params_grid = np.mgrid[5:12.1:0.6, 7:13.1:0.6].reshape(2,-1).T

params_distrib = estim_params(M, dist_mtx, x_grid, params_grid, 0.05)

# Plot the best couples of parameters
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

params_distrib = estim_params(M[:,:2], dist_mtx[:2,:2], x_grid, params_grid, 0.0001)

# Plot the best couples of parameters    
v,c = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(v, c, bins=(4, 4), cmap=plt.cm.BuPu)

# Parameters Marginal densities
pd.Series(v).plot("kde")
pd.Series(c).plot("kde")



#============================================================
# Matern-Whithney kernel: Try with 5 locations
#============================================================
d = 5

min_M_emp = np.min(M,axis=0)[:d]
max_M_emp = np.max(M,axis=0)[:d]

# Simulate the grid over which F and hat_F will be evaluated
#x_grid = np.mgrid[min_M_emp[i]:max_M_emp[i]+0.1:0.2 for i in range(d)].reshape(2,-1).T
x_grid = np.mgrid[min_M_emp[0]:max_M_emp[0]+0.1:1.2,\
                  min_M_emp[1]:max_M_emp[1]+0.1:1.2,\
                  min_M_emp[2]:max_M_emp[2]+0.1:1.2,\
                  min_M_emp[3]:max_M_emp[3]+0.1:1.2,\
                  min_M_emp[4]:max_M_emp[4]+0.1:1.2].reshape(d,-1).T
                  
params_grid = np.mgrid[0:20.1:5, 100:300.1:12].reshape(2,-1).T
params_distrib = estim_params(M[:,:d], dist_mtx[:d,:d], x_grid, params_grid, 0.10)

# Plot the best couples of parameters    
v,c = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(v, c, bins=(4, 4), cmap=plt.cm.BuPu)

# Parameters Marginal densities
pd.Series(v).plot("kde")
pd.Series(c).plot("kde")


#===================================================================
# Matern-Whithney kernel: Results variance analysis with 2 locations
#===================================================================
d = 2

min_M_emp = np.min(M,axis=0)[:d]
max_M_emp = np.max(M,axis=0)[:d]

NB_RUNS = 30
########### Fix grid
x_grid = np.mgrid[min_M_emp[0]:max_M_emp[0]+0.1:0.4, min_M_emp[1]:max_M_emp[1]+0.1:0.4].reshape(2,-1).T
params_grid = np.mgrid[19.5:22.5:0.4, 160:260.1:5].reshape(2,-1).T

dists_multiple_runs = []
for j in range(NB_RUNS):
    print(j,"th run !")
    dists_multiple_runs.append(estim_params(M[:,:d], dist_mtx[:d,:d], x_grid, params_grid, 0.0001, full_output=True))
    print("-------------------------------------------")

intrarun_var = []
for j in range(NB_RUNS):
    intrarun_var.append(np.var(list(dists_multiple_runs[j].values())))

avg_intrarun_var = np.mean(intrarun_var)
interrun_var = np.var([min(dists_multiple_runs[j].values()) for j in range(10)])

print(interrun_var/avg_intrarun_var) 

########### Monte Carlo grid
x_grid = np.random.uniform(min_M_emp[:d]+1,max_M_emp[:d]+5, size=(154,d))
params_grid = np.mgrid[19.5:22.5:0.4, 160:260.1:5].reshape(2,-1).T

dists_multiple_runs = []
for j in range(NB_RUNS):
    print(j,"th run !")
    dists_multiple_runs.append(estim_params(M[:,:d], dist_mtx[:d,:d], x_grid, params_grid, 0.0001, full_output=True, verbose=False))
    print("-------------------------------------------")

intrarun_var = []
for j in range(NB_RUNS):
    intrarun_var.append(np.var(list(dists_multiple_runs[j].values())))

avg_intrarun_var = np.mean(intrarun_var)
interrun_var = np.var([min(dists_multiple_runs[j].values()) for j in range(10)])

print(interrun_var/avg_intrarun_var) # Intrarun variance is 10 times higher than interrun variance: good sign


#===================================================================
# Matern-Whithney kernel: Results variance analysis with 5 locations
#===================================================================
d = 5

min_M_emp = np.min(M,axis=0)[:d]
max_M_emp = np.max(M,axis=0)[:d]

NB_RUNS = 30
########### Fix grid
x_grid = np.mgrid[min_M_emp[0]:max_M_emp[0]+0.1:1.5,\
                  min_M_emp[1]:max_M_emp[1]+0.1:1.5,\
                  min_M_emp[2]:max_M_emp[2]+0.1:1.5,\
                  min_M_emp[3]:max_M_emp[3]+0.1:1.5,\
                  min_M_emp[4]:max_M_emp[4]+0.1:1.5].reshape(d,-1).T
                  
x_grid.shape
                  
params_grid = np.mgrid[19.5:22.5:0.4, 160:260.1:5].reshape(2,-1).T

dists_multiple_runs = []
for j in range(NB_RUNS):
    print(j,"th run !")
    dists_multiple_runs.append(estim_params(M[:,:d], dist_mtx[:d,:d], x_grid, params_grid, 0.0001, full_output=True, verbose=False))
    print("-------------------------------------------")

intrarun_var = []
for j in range(NB_RUNS):
    intrarun_var.append(np.var(list(dists_multiple_runs[j].values())))

avg_intrarun_var = np.mean(intrarun_var)
interrun_var = np.var([min(dists_multiple_runs[j].values()) for j in range(NB_RUNS)])

print(interrun_var/avg_intrarun_var) 

########### Monte Carlo grid
NB_RUNS = 7
x_grid = np.random.uniform(min_M_emp[:d]+1,max_M_emp[:d]+5, size=(960,d))
params_grid = np.mgrid[19.5:22.5:0.4, 160:260.1:5].reshape(2,-1).T

dists_multiple_runs = []
for j in range(NB_RUNS):
    print(j,"th run !")
    dists_multiple_runs.append(estim_params(M[:,:d], dist_mtx[:d,:d], x_grid, params_grid, 0.0001, full_output=True, verbose=False))
    print("-------------------------------------------")

intrarun_var = []
for j in range(7):
    intrarun_var.append(np.var(list(dists_multiple_runs[j].values())))

avg_intrarun_var = np.mean(intrarun_var)
interrun_var = np.var([min(dists_multiple_runs[j].values()) for j in range(NB_RUNS)])

print(interrun_var/avg_intrarun_var) # Intrarun variance is 10 times higher than interrun variance: good sign



#====================================================================================================
# Matern-Whithney kernel: Ploting the estimated cdf for 2 locations (do not consider for the moment)
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
# Testing the approach with a simpler model (ok for 2D gaussian)
#====================================================================================================
from scipy.stats import multivariate_normal

d = 2
nb_points = 200
#x_grid = np.mgrid[-4:4:1, -4:4+0.1:1].reshape(2,-1).T # 72 points to few
x_grid = np.random.uniform(-4+1,4+2, size=(1500,d))

#x_grid = np.linspace(-4,4,30) # 1D recover without probs

params_grid = np.mgrid[0.1:3.1:0.5, 0.1:3.1:0.5].reshape(2,-1).T # Ok for mean and var estimation
params_grid.shape
true_param = (1,2)

#X_true = np.random.normal(size=50000, loc=true_param)
cdf_dist = {}
    

for param in params_grid:
    print(param)
    cdf_dist[str(param)] = []
    X_param = np.random.multivariate_normal(size=nb_points,mean=np.full(d,0), cov=np.diag(param))

    for x in x_grid: 
        # Compute the theoretical and empirical cdfs
        emp_cdf = cdf(X_param,x) #sum(X_param<=x)/len(X_param)
        th_cdf = multivariate_normal.cdf(x, cov=np.diag(true_param))

        # Store the squared euclidian distance for all x and the current params
        current_dist = (emp_cdf-th_cdf)**2
        cdf_dist[str(param)].append(current_dist)
        
# Compute the mean distance for each parameters couple
for params in params_grid:
    cdf_dist[str(params)] = np.mean(cdf_dist[str(params)])

# Define a threshold to keep only the 5% of the lowest distance associated parameters
accept_threshold = np.quantile(list(cdf_dist.values()), 0.005)

params_distrib = []
for params in params_grid:
    if cdf_dist[str(params)]<accept_threshold:
        params_distrib.append(params)
params_distrib = np.stack(params_distrib)
params_distrib


#==================================================================================
# Testing the approach with a simpler model (ok for 2D gaussian) for several runs
#==================================================================================

d = 2
nb_points = 20
#x_grid = np.mgrid[-4:4:1, -4:4+0.1:1].reshape(2,-1).T # 72 points to few
x_grid = np.random.uniform(-4+1,4+2, size=(1500,d))

#x_grid = np.linspace(-4,4,30) # 1D recover without probs

params_grid = np.mgrid[0.1:3.1:0.5, 0.1:3.1:0.5].reshape(2,-1).T # Ok for mean and var estimation
params_grid.shape
true_param = (1,2)

#X_true = np.random.normal(size=50000, loc=true_param)
cdf_dist = {}
    

for param in params_grid:
    print(param)
    cdf_dist[str(param)] = []
    X_param = np.random.multivariate_normal(size=nb_points,mean=np.full(d,0), cov=np.diag(param))

    for x in x_grid: 
        # Compute the theoretical and empirical cdfs
        emp_cdf = cdf(X_param,x) #sum(X_param<=x)/len(X_param)
        th_cdf = multivariate_normal.cdf(x, cov=np.diag(true_param))

        # Store the squared euclidian distance for all x and the current params
        current_dist = (emp_cdf-th_cdf)**2
        cdf_dist[str(param)].append(current_dist)
        
# Compute the mean distance for each parameters couple
for params in params_grid:
    cdf_dist[str(params)] = np.mean(cdf_dist[str(params)])

# Define a threshold to keep only the 5% of the lowest distance associated parameters
accept_threshold = np.quantile(list(cdf_dist.values()), 0.005)

params_distrib = []
for params in params_grid:
    if cdf_dist[str(params)]<accept_threshold:
        params_distrib.append(params)
params_distrib = np.stack(params_distrib)
params_distrib




#====================================================================================================
# Fixing one param and do dyadic search
#====================================================================================================
v_hat = []
for i in range(20):
    print(i,"th run")
    v_hat.append(dyadic_param_search(M, 8, dist_mtx))

v_hat_loc67 = np.stack(v_hat)
pd.Series(v_hat_loc67[:,1]).plot('kde')


v_hat_loc01 = np.stack(v_hat)
pd.Series(v_hat_loc01[:,1]).plot('kde')

plt.hist(v_hat_loc67[:,1], color = 'red', edgecolor = 'black',
bins = int(180/5))
plt.title('Results of 50 dyadic-like estimations of the c parameter for localisations 6 and 7')
plt.xlabel("Estimated value")
plt.ylabel("Times it has been estimated")


plt.hist(v_hat_loc01[:,1], color = 'blue', edgecolor = 'black',
bins = int(180/5))
plt.title('Results of 50 dyadic-like estimations of the c parameter for localisations 0 and 1')
plt.xlabel("Estimated value")
plt.ylabel("Times it has been estimated")

d = 2

min_M_emp = np.min(M,axis=0)[:2]
max_M_emp = np.max(M,axis=0)[:2]

n_iter = 100
c_fixed = 5
epsilon = 3
nb_values_tested = 20

SUP = 300.1
INF = 2
interv_len = SUP-INF


# Initialisation
x_grid = np.random.uniform(min_M_emp[:d]+1,max_M_emp[:d]+2, size=(800,d))
params_grid = np.mgrid[c_fixed:c_fixed+0.1:0.4, INF:SUP:interv_len/nb_values_tested].reshape(2,-1).T

while((SUP-INF>epsilon) and n_iter>0):
    print("Inf=",INF, "SUP=", SUP)
    params_distrib = estim_params(M[:,:2], dist_mtx[:2,:2], x_grid, params_grid, full_output=True, threshold=1, verbose=False)
    INF, SUP = find_inf_sup_search(params_distrib) # Get the two best candidates and search in this region
    interv_len = SUP-INF # Legnth of the interval in which we are searching
    
    # Simulate the grid over which F and hat_F will be evaluated
    x_grid = np.random.uniform(min_M_emp[:d]+1,max_M_emp[:d]+2, size=(300,d))
    params_grid = np.mgrid[c_fixed:c_fixed+0.1:0.4, INF:SUP:interv_len/nb_values_tested].reshape(2,-1).T
    
    nb_values_tested= max(3, nb_values_tested-7) # Test a lot of position at the beginning and fewer at the end
    n_iter-=1
    

# Final estimation 
best_param = estim_params(M[:,:2], dist_mtx[:2,:2], x_grid, params_grid, threshold=0.00001, verbose=False)[0]
print("The best parameter found for (c,v) was", best_param)


#============================================================
# Matern-Whithney kernel: 1 parameter Try with 2 locations
#============================================================
d = 2

min_M_emp = np.min(M,axis=0)[:2]
max_M_emp = np.max(M,axis=0)[:2]

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.mgrid[min_M_emp[0]:max_M_emp[0]+0.1:0.2, min_M_emp[1]:max_M_emp[1]+0.1:0.2].reshape(2,-1).T
params_grid = np.mgrid[5:5.1:0.4, :260.1:2].reshape(2,-1).T

params_distrib = estim_params(M[:,:2], dist_mtx[:2,:2], x_grid, params_grid, 0.0001)

# Plot the best couples of parameters    
v,c = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(v, c, bins=(4, 4), cmap=plt.cm.BuPu)

# Parameters Marginal densities
pd.Series(v).plot("kde")
pd.Series(c).plot("kde")


#====================================================================================================
# Matern-Whithney kernel: Testing if refind the true parameter value
#====================================================================================================

### Simulating M for 2 locations
d = 2
t = [1,8]
dist_mtx = dist_matrix(t)

# Generate copies of M for a given couple of parameters
#M_simu = []
for i in range(10000):
    if (i%1000==0):
        print(i)
    M_simu.append(simul_M(100000, rho(dist_mtx, 1,10)))

M_simu_np = np.stack(M_simu)

non_inf_idx = np.all(M_simu_np!=np.inf,axis=1) # Problem of A1=0
clean_M_simu = M_simu_np[non_inf_idx, :]
pd.DataFrame(clean_M_simu).to_csv("clean_simu_M1_10_12000.csv", index=False)

clean_M_simu.shape

# Simulating 
M_1_10 = pd.read_csv("clean_simu_M1_10_1200.csv").values

d = 2

min_M_emp = np.min(M_1_10,axis=0)
max_M_emp = np.max(M_1_10,axis=0)

# Simulate the grid over which F and hat_F will be evaluated
x_grid = np.random.uniform(min_M_emp+1,max_M_emp+5, size=(3000,d))
params_grid = np.mgrid[1.0:1.01:0.4, 1:20.1:3].reshape(2,-1).T
params_grid.shape

computed_dists = estim_params(M_1_10, dist_mtx, x_grid, params_grid, 0.3, full_output=True)

# Plot the best couples of parameters    
v,c = params_distrib[:,0], params_distrib[:,1]
plt.hist2d(v, c, bins=(4, 4), cmap=plt.cm.BuPu)

# Parameters Marginal densities
pd.Series(v).plot("kde")
pd.Series(c).plot("kde")


