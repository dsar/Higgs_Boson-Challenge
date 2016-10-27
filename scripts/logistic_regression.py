# -*- coding: utf-8 -*-

import numpy as np

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

def logistic_regression_gradient_descent(y, x, max_iter=4000, threshold=1e-8, gamma = 0.000000001, print_=True):
    # init parameters
    losses = []

    # build tx
    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    tx = x
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
