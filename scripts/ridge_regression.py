# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from proj1_helpers import score,build_k_indices, kfold_split_data,split_data
from costs import compute_loss, compute_rmse
from build_polynomial import build_poly, find_best_poly, build_optimal
from plots import cross_validation_visualization, bias_variance_decomposition_visualization

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    N = tx.shape[0]
    D = tx.shape[1]
    
    ident = np.eye(D) * lamb * 2 * N
    w = np.linalg.solve(np.transpose(tx).dot(tx) + ident, np.transpose(tx).dot(y))

    loss = compute_loss(y, tx, w)

    return w, loss

def cross_validation(y, tx, k_indices, k, lambda_):
    """return the loss of ridge regression."""

    loss_tr=np.zeros(k)
    loss_te=np.zeros(k)
    for i in range(k):
        test_indices = k_indices[i]
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        train_indices = np.array([index for indices in k_indices[mask] for index in indices])

        w, loss = ridge_regression(y[train_indices], tx[train_indices], lambda_)

        loss_tr[i] = compute_rmse(loss)
        loss_te[i] = compute_rmse(compute_loss(y[test_indices], tx[test_indices], w))
    
    return np.average(loss_tr), np.average(loss_te)

def get_best_lambda(lambdas, rmse_tr, rmse_te):
	min_rmse = np.min(rmse_te)
	print(min_rmse)
	index = rmse_te.index(min_rmse)
	return lambdas[index]

def cross_validation_ridge_regression(y,x,seed=1, k_fold=5, lambdas=None):

    if lambdas == None:
        lambdas = np.logspace(-15, 3, 20)

    k_indices = build_k_indices(y, k_fold, seed)
    rmse_tr = []
    rmse_te = []

    for lambda_ in lambdas:
        print('lambda_: ',lambda_)
        tx = x[:,1:]
        best_deg = find_best_poly(y, tx, test_RR)
        opt_tr = build_optimal(tx, best_deg)
        # w, _ = ridge_regression(y, opt_tr,0.01)
        loss_tr, loss_te = cross_validation(y, opt_tr, k_indices, k_fold, lambda_)
        rmse_tr.append(np.copy(loss_tr))
        rmse_te.append(np.copy(loss_te))

    cross_validation_visualization(lambdas, rmse_tr, rmse_te)

    return min(rmse_te), get_best_lambda(lambdas, rmse_tr, rmse_te)


# Ridge regression
def test_RR(y, tX, lambda_=0.01, ratio=0.2,seed = 1):
    y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)
    w, _ = ridge_regression(y_train, x_train, lambda_)
    s = score(x_test, y_test, w)
    return s
