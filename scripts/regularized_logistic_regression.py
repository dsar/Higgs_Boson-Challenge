# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import build_k_indices, kfold_split_data, split_data, logistic_score
from costs import compute_loss
from build_polynomial import build_poly, find_best_poly, build_optimal
from logistic_regression import sigmoid
from plots import cross_validation_visualization

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    y = y.reshape((y.shape[0], 1))
    total_loss = (1.0/y.shape[0])* np.sum(np.log(1+np.exp(tx.dot(w))) - y*tx.dot(w))
    return total_loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    y = y.reshape((y.shape[0], 1))
    gradient = np.transpose(tx).dot(sigmoid(tx.dot(w))-y)
    return gradient

def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss_regulirazer = np.asscalar(lambda_ * (np.transpose(w).dot(w)))
    gradient_regulirazer = 2 * lambda_ * w
    
    loss = calculate_loss(y, tx, w) + loss_regulirazer
    gradient = calculate_gradient(y, tx, w) + gradient_regulirazer
    w = w - gamma * gradient
    return loss, w

def regularized_logistic_regression_gradient_descent(y, tx, max_iter=4000, threshold=1e-8, gamma = 0.000000001, lambda_=0.01, print_=True):
    # init parameters
    losses = []

    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma,lambda_)
        # log info
        if iter % 1000 == 0:
            if print_ == True:
                print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss, w

def rlr_cross_validation(y, tx, k_indices, k, lambda_):
    """return the loss of ridge regression."""

    loss_tr=np.zeros(k)
    loss_te=np.zeros(k)
    for i in range(k):
        test_indices = k_indices[i]
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        train_indices = np.array([index for indices in k_indices[mask] for index in indices])

        loss, w = regularized_logistic_regression_gradient_descent(y[train_indices], tx[train_indices], lambda_=lambda_)

        loss_tr[i] = compute_loss(y[train_indices],tx[train_indices],w)
        loss_te[i] = compute_loss(y[test_indices], tx[test_indices], w)
    
    return np.average(loss_tr), np.average(loss_te)

def get_best_lambda(lambdas, mse_tr, mse_te):
    min_mse = np.min(mse_te)
    index = mse_te.index(min_mse)
    return lambdas[index]

def cross_validation_regularized_logistic_regression(y,x,seed=1, k_fold=5, lambdas=None):

    if lambdas == None:
        #be careful with negative values of loss
        lambdas = np.logspace(-15, 3, 5)

    k_indices = build_k_indices(y, k_fold, seed)
    mse_tr = []
    mse_te = []

    for lambda_ in lambdas:
        print('lambda: ',lambda_)
        tx = x[:,1:]
        best_deg = find_best_poly(y, tx, test_RLR)
        opt_tr = build_optimal(tx, best_deg)
        loss_tr, loss_te = rlr_cross_validation(y, opt_tr, k_indices, k_fold, lambda_=lambda_)
        mse_tr.append(np.copy(loss_tr))
        mse_te.append(np.copy(loss_te))

    cross_validation_visualization(lambdas, mse_tr, mse_te)

    return min(mse_te), get_best_lambda(lambdas, mse_tr, mse_te)

# Regularized logistic regression
def test_RLR(y, tX, gamma=0.000000001,ratio=0.2, seed=1, print_=False, threshold=0.5):
    y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

    loss, w = regularized_logistic_regression_gradient_descent(y, tX,print_=False)
    s = logistic_score(x_test,y_test,w,threshold)
    return s