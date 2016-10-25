# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np
from math import floor


# def split_data(x, y, ratio, seed=1):
#     """split the dataset based on the split ratio."""
#     # set seed
#     np.random.seed(seed)
    
#     N = y.shape[0]

#     indices = np.arange(N)
#     np.random.shuffle(indices)
#     x = x[indices]
#     y = y[indices]
    
#     return x[:int(N*ratio)], x[int(N*ratio):], y[:int(N*ratio)], y[int(N*ratio):]


def split_data(y, X, ratio, seed):
	np.random.seed(seed)
	combined = np.c_[X, y]
	np.random.shuffle(combined)
	split = floor(ratio * X.shape[0])
	X1 = combined[:split, :X.shape[1]]
	y1 = combined[:split, X.shape[1]:]
	X2 = combined[split:, :X.shape[1]]
	y2 = combined[split:, X.shape[1]:]
	return X1, y1, X2, y2

