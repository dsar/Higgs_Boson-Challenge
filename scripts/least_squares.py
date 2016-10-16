# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import *


def least_squares(y, tx):
    """calculate the least squares solution."""
    # least squares: TODO
    # returns mse, and optimal weights
    
    # problem with linalg.inv
#     w_opt = np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx)).dot(y)
    w_opt = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
    mse = compute_loss(y, tx, w_opt)
    
    return w_opt, mse