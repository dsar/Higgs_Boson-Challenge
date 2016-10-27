# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(tX, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""

    tmp = tX[:,1:]
    # print(tmp.shape)
    for d in range(2,degree+1):
        tmp = np.append(tmp,np.power(tX[:,1:],d),axis=1)

    tmp = np.append(np.ones((tmp.shape[0],1)),tmp,axis=1)

    return tmp


# def build_poly(tX, degree):
# """polynomial basis functions for input data x, for j=0 up to j=degree."""

	# N = x.shape[0]
	# φ = np.ones([N, degree+1])

	# for d in range(1,degree+1):
	#     φ[:,d] = x**d
		
	# return φ