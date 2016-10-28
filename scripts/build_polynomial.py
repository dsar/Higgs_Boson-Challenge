# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

# UPDATED
# def build_poly(tX, degree):
#     """polynomial basis functions for input data x, for j=0 up to j=degree."""

#     tmp = tX[:,1:]
#     # print(tmp.shape)
#     for d in range(2,degree+1):
#         tmp = np.append(tmp,np.power(tX[:,1:],d),axis=1)

#     tmp = np.append(np.ones((tmp.shape[0],1)),tmp,axis=1)

#     return tmp


def build_poly(tX, degree):
    N = tX.shape[0]
    φ = np.ones([N, degree+1])
    for d in range(1,degree+1):
        φ[:,d] = tX**d
    return φ



def find_best_poly(y, tX, test_function, max_degree=5):
	best_degrees = np.ones(tX.shape[1], dtype=np.int8)
	best_scores = np.zeros(tX.shape[1])
	for i in range(tX.shape[1]):
		for d in range(max_degree+1):
			poly = build_poly(tX[:,i], d)
			old = np.delete(tX, i, axis=1)
			full = np.c_[old, poly]
			score = test_function(y, full)
			if score > best_scores[i]:
				best_scores[i] = score
				best_degrees[i] = d
	print(np.c_[best_degrees, best_scores])
	return best_degrees

def build_optimal(tX, best_degrees):
	opt = np.ones(tX.shape[0])
	for i in range(tX.shape[1]):
		poly = build_poly(tX[:,i], best_degrees[i])
		opt = np.c_[opt, poly[:,1:]]
	print(opt.shape)
	return opt
	#score = test_LS(y, opt)
	#print(score)
