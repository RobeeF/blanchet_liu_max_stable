# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 16:34:41 2019

@author: robin
"""

import numpy as np
import math

#===========================================================
# Estimation part
#===========================================================

def compute_bar_Wn(x,n,M,X,cov):
    d = cov.shape[0]
    w_d= math.pi**(float(d)/2)/math.gamma(float(d)/2 + 1)
    delta_n = 1/np.log(np.log(np.log(n+np.exp(np.exp(1)))))
    numerator = np.dot((M-x),np.dot(np.linalg.inv(cov),X.T).sum(axis=1))
    denominator = d*w_d*((np.linalg.norm(x-M, ord=2))**d + delta_n*np.linalg.norm(x-M, ord=2))
    return numerator/denominator


def g(n):
    return 1/n*np.log(n+np.exp(1)-1)*np.log(np.log(n+np.exp(np.exp(1))-1))

def compute_V_x(x,M,X,cov,L):
    V_k = []
    for k in range(1,L+1):   
        V_k.append((compute_bar_Wn(x,k,M,X,cov)- compute_bar_Wn(x,k-1,M,X,cov))/g(k))
    return np.sum(V_k)
