# -*- coding: utf-8 -*-
import numpy as np
from helpers import build_poly, score
from implementations import least_squares

def best_feature_degrees(y, tx, test_function, max_degree=5):
	"""Compute the optimal polynomial expansion degree for each feature individually"""
	D = tx.shape[1]
	best_degrees = np.ones(D, dtype=np.int8)
	best_scores = np.zeros(D)

	for i in range(D):
		for d in range(max_degree+1):
			feature_poly = build_poly(tx[:,i], d)
			tx_no_i = np.delete(tx, i, axis=1)
			feature_exp = np.c_[tx_no_i, feature_poly]

			w, _ = test_function(y, feature_exp)
			s = score(y, feature_exp, w)
			if s > best_scores[i]:
				best_scores[i] = s
				best_degrees[i] = d

	return best_degrees

def build_poly_by_feature(tx, degrees):
	"""Builds a polynomial expansion based on the given degree for each feature"""
	N = tx.shape[0]
	D = tx.shape[1]

	exp = np.ones(N)
	for i in range(D):
		feature_poly = build_poly(tx[:,i], degrees[i])
		exp = np.c_[exp, feature_poly[:,1:]]
	
	return exp
