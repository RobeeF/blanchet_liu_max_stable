# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:27:33 2019

@author: robin
"""

import os

os.chdir('C:/Users/robin/Documents/GitHub/test')

from simulate_M import *
from estimator import *


#===========================================================
# Define the parameters of the model
#===========================================================
d=3
sigma_square = np.array([1,2,4])
cov = np.diag(sigma_square)
a = compute_a()

#===========================================================
# Compute M, X, N and the estimator
#===========================================================

M,X,N = algorithm_M(a, cov)
L =100 # Taken as a deterministic value for the moment
V = compute_V_x((1/3,2/3,1), M, X, cov, 10000)
print(V)

