# -*- coding: utf-8 -*-

import numpy as np
from costs import compute_loss


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # TODO: compute gradient and loss
    e = y - (np.dot(tx,w))
    N = y.shape[0]
    
    gradient = -(1/N) * np.dot(np.transpose(tx),e)
    loss = compute_loss(y,tx,w)
    
    return gradient, loss


def least_squares_GD(y, tx, initial_w, gamma, max_iters): 
    """Gradient descent algorithm."""
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

        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    print('parameters w: ',ws[-1])

    return losses, ws