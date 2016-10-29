# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import build_k_indices, kfold_split_data, split_data, logistic_score
from costs import compute_loss
from build_polynomial import build_poly, find_best_poly, build_optimal

"""NEWTON METHOD"""
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate hessian: TODO
    # ***************************************************
    # N = y.shape[0]
    # S = np.zeros((N,N))
    # for n in range(N):
    #     S[n,n] = sigmoid(np.transpose(tx[n]).dot(w))*(1-sigmoid(np.transpose(tx[n]).dot(w)))
    # return np.transpose(tx).dot(S).dot(tx)

    S = np.diag((sigmoid(tx.dot(w)) * (1 - sigmoid(tx.dot(w))))[:,0])
    H = np.transpose(tx).dot(S).dot(tx)
    
    return H 

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and hessian: TODO
    # ***************************************************
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
    H = calculate_hessian(y,tx,w)
    return loss, gradient, H

def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient and hessian: TODO
    # ***************************************************
    loss, gradient, H = logistic_regression(y,tx,w)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    w = w - gamma * np.linalg.inv(H).dot(gradient)
    return loss, w