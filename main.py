# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:27:33 2019

@author: Robin Fuchs
"""

import os

os.chdir('C:/Users/robin/Documents/GitHub/blanchet_liu_max_stable')

from estimator import compute_f_hat_b
import numpy as np

#===========================================================
# Define the parameters of the model
#===========================================================
t = [1/3,2/3,1]
dist = [1/3,1/3,1/3]
d=3
cov = cov_brownian_X(dist) # For standard brownian motion
    
#===========================================================
# Compute M, X, N and the estimation of the density
#===========================================================

# Computational budget setting:
b = 10**2
# Takes forever to run because of infinite-like loop:
# The estimation is also erroneous certainly due to delta and theta
compute_f_hat_b(t, b, cov, conf_lvl=.05)
# check X is not of dim 1


