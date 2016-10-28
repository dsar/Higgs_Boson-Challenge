import numpy as np
from build_polynomial import build_poly

from test import *

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
