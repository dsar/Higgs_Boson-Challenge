# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""

import numpy as np
from helpers import batch_iter
from gradient_descent import compute_gradient
from costs import compute_loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    # TODO: implement stochastic gradient computation.

    gradient,loss = compute_gradient(y, tx, w)
    return gradient

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # TODO: implement stochastic gradient descent.

    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter, (minibatch_y, minibatch_tx) in zip(range(max_iters), batch_iter(y, tx, batch_size)):
        
        stoch_gradient = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
        loss = compute_loss(y,tx,w)
        
        w = w - gamma * stoch_gradient

        ws.append(np.copy(w))
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
