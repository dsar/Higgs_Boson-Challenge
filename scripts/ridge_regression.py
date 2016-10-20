# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    N = tx.shape[0]
    D = tx.shape[1]
    
    ident = np.eye(D)*lamb*2*N
    return np.linalg.solve(np.transpose(tx).dot(tx)+ident, np.transpose(tx).dot(y))
