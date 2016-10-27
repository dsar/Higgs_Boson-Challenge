# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import build_k_indices, kfold_split_data, split_data, logistic_score
from costs import compute_loss
from build_polynomial import build_poly, find_best_poly, build_optimal

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))

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

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return loss, w

def logistic_regression_gradient_descent(y, tx, max_iter=4000, threshold=1e-8, gamma = 0.000000001, print_=True):
    # init parameters
    losses = []

    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 1000 == 0:
            if print_ == True:
                print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return loss, w

def lr_cross_validation(y, tx, k_indices, k, gamma):
    """return the loss of ridge regression."""

    loss_tr=np.zeros(k)
    loss_te=np.zeros(k)
    for i in range(k):
        test_indices = k_indices[i]
        mask = np.ones(k, dtype=bool)
        mask[i] = False
        train_indices = np.array([index for indices in k_indices[mask] for index in indices])

        loss, w = logistic_regression_gradient_descent(y[train_indices], tx[train_indices], gamma=gamma)

        loss_tr[i] = compute_loss(y[train_indices],tx[train_indices],w)
        loss_te[i] = compute_loss(y[test_indices], tx[test_indices], w)
    
    return np.average(loss_tr), np.average(loss_te)

def get_best_gamma(gammas, mse_tr, mse_te):
    min_mse = np.min(mse_te)
    index = mse_te.index(min_mse)
    return gammas[index]

def cross_validation_logistic_regression(y,x,seed=1, k_fold=5, gammas=None):

    if gammas == None:
        #be careful with negative values of loss
        gammas = np.logspace(-13, -7, 5)

    k_indices = build_k_indices(y, k_fold, seed)
    mse_tr = []
    mse_te = []

    for gamma in gammas:
        print('gamma: ',gamma)
        tx = x[:,1:]
        best_deg = find_best_poly(y, tx, test_LR)
        opt_tr = build_optimal(tx, best_deg)
        loss_tr, loss_te = lr_cross_validation(y, opt_tr, k_indices, k_fold, gamma)
        mse_tr.append(np.copy(loss_tr))
        mse_te.append(np.copy(loss_te))

    # cross_validation_visualization(lambdas, mse_tr, mse_te)

    min_mse = min(mse_te)
    if min_mse < 0:
        min_mse = 1000000000
    return min_mse, get_best_gamma(gammas, mse_tr, mse_te)

# Logistic regression
def test_LR(y, tX, gamma=0.000000001,ratio=0.2, seed=1, print_=False, threshold=0.5):
    y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

    loss, w = logistic_regression_gradient_descent(y, tX,print_=False)
    s = logistic_score(x_test,y_test,w,threshold)
    return s