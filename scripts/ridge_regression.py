# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from costs import *


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    N = tx.shape[0]
    D = tx.shape[1]

    w_ridge = np.linalg.inv(np.transpose(tx).dot(tx) + lamb*2*N*np.identity(D)).dot(np.transpose(tx)).dot(y)

    loss = compute_loss(y, tx, w_ridge)

    print('loss: ',loss)

    rmse = np.sqrt(2*compute_loss(y, tx, w_ridge))

    print('rmse: ',rmse)

    print('parameters: ',w_ridge)

    return w_ridge
    
