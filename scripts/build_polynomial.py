# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    N = x.shape[0]
    φ = np.ones([N, degree+1])
    
    for d in range(1,degree+1):
        φ[:,d] = x**d
        
    return φ
