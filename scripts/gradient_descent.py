# -*- coding: utf-8 -*-

import numpy as np
from costs import compute_loss
from proj1_helpers import split_data, score


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # TODO: compute gradient and loss
    e = y - (np.dot(tx,w))
    N = y.shape[0]
    
    gradient = -(1/N) * np.dot(np.transpose(tx),e)
    loss = compute_loss(y,tx,w)
    
    return gradient, loss


def least_squares_GD(y, tx, initial_w, gamma, max_iters, print_=True): 
    """Gradient descent algorithm."""

    if print_ == True:
        print('y shape: ',y.shape)
        print('tx shape: ',tx.shape)

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        # TODO: compute gradient and loss
        gradient, loss = compute_gradient(y,tx,w)
        
        # TODO: update w by gradient
        w = w - gamma * gradient
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

        if print_ == True:
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss))

    print('parameters w: ',ws[-1])

    return losses, ws

# Gradient descent
def test_GD(y, tX, ratio=0.2, w_initial=None, gamma=0.01, max_iters=1000, seed=1):
    y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

    features = x_train.shape[1]
    
    if w_initial == None:
        w_initial = np.zeros(features)

    _, w = least_squares_GD(y_train, x_train, w_initial, gamma, max_iters ,print_=False)

    s = score(x_test, y_test, w[-1])

    return s