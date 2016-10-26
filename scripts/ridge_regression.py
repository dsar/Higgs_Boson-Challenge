# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from helpers import build_k_indices, kfold_split_data
from costs import compute_loss
from build_polynomial import build_poly
from plots import cross_validation_visualization

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    N = tx.shape[0]
    D = tx.shape[1]
    
    ident = np.eye(D)*lamb*2*N
    return np.linalg.solve(np.transpose(tx).dot(tx) + ident, np.transpose(tx).dot(y))

def cross_validation_step(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    N = y.shape[0]

    test_indices = k_indices[k]
    train_indices = [item for item in range(N) if item not in test_indices]
    
    # φ_x = build_poly(x, degree)
    φ_x = x
    x_test,x_train,y_test,y_train = kfold_split_data(φ_x, y,test_indices)
    
    w_ridge = ridge_regression(y_train, x_train, lambda_)
    
    rmse_train = np.sqrt(2*compute_loss(y_train, x_train, w_ridge))
    rmse_test = np.sqrt(2*compute_loss(y_test, x_test, w_ridge))
    
    return rmse_train, rmse_test

# def cross_validation(y, x, k_indices, k, lambda_, degree):
#     """return the loss of ridge regression."""

#     x_test = x[k_indices[k]]
#     y_test = y[k_indices[k]]
#     x_train = np.delete(x, k_indices[k])
#     y_train = np.delete(y, k_indices[k])
    
#     assert(len(x_test) == int(len(x)/len(k_indices)))
    
#     x_test = build_poly(x_test, degree)
#     x_train = build_poly(x_train, degree)
    
    
#     weights = ridge_regression(y_train, x_train, lambda_)
    
#     loss_tr = compute_mse(y_train, x_train, weights)
#     loss_te = compute_mse(y_test, x_test, weights)
#     return loss_tr, loss_te

def cross_validation_ridge_regression(y,x,seed=1, degree=7, k_fold=5, lambdas=None):
	if lambdas == None:
		lambdas = np.logspace(-4, 0, 30)
	# split data in k fold
	k_indices = build_k_indices(y, k_fold, seed)
	# define lists to store the loss of training data and test data
	mse_tr = [0]*len(lambdas)
	mse_te = [0]*len(lambdas)
	# ***************************************************
	# INSERT YOUR CODE HERE
	# cross validation: TODO
	# ***************************************************
	for k in range(k_fold):
	    i = 0
	    for lamb in lambdas:
	        tmp_tr, tmp_te = cross_validation_step(y, x, k_indices, k, lambdas, degree)
	        mse_tr[i] = tmp_tr + mse_tr[i]
	        mse_te[i] = tmp_te + mse_te[i]
	        i = i+1

	mse_tr = [(x/k_fold) for x in mse_tr]
	mse_te = [(x/k_fold) for x in mse_te]
	cross_validation_visualization(lambdas, mse_tr, mse_te)


