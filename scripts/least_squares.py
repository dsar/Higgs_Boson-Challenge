# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import *
from proj1_helpers import split_data, score


def least_squares(y, tx):
    """calculate the least squares solution."""
    # least squares: TODO
    # returns mse, and optimal weights
        
    w_opt = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    mse = compute_loss(y, tx, w_opt)
    
    return w_opt, mse

# Least squares
def test_LS(y,tX,ratio=0.2,seed=1):
	y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

	w, _ = least_squares(y_train, x_train)
	s = score(x_test, y_test, w)
	
	return s