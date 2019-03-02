# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:27:33 2019

@author: Robin Fuchs
"""

import os

os.chdir('C:/Users/robin/Documents/GitHub/blanchet_liu_max_stable')

from simulate_M import compute_a
from estimator import simulate_L, compute_V_x, compute_f_hat_b
import numpy as np
import pandas as pd

#===========================================================
# Define the parameters of the model
#===========================================================
d=3
cov = np.identity(d) # For standard brownian motion
    
#===========================================================
# Compute M, X, N and the estimation of the density
#===========================================================
x = np.full(d,0) # Evaluate the density for x=(0,0,0)

# Computational budget setting:
b = 10**2
# Takes forever to run because of infinite-like loop:
# The estimation is also erroneous certainly due to delta and theta
compute_f_hat_b(x, b, cov, conf_lvl=.05)



