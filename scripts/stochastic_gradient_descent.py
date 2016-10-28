# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import batch_iter,split_data, score
from gradient_descent import compute_gradient
from costs import compute_loss

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    # TODO: implement stochastic gradient computation.

    gradient,loss = compute_gradient(y, tx, w)
    return gradient

def least_squares_SGD(y, tx, initial_w, batch_size, gamma, max_iters, print_=True):
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
        if print_ == True:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    print('parameters w: ',ws[-1])

    return losses, ws

def get_min_param_index(sgd_losses):
    index = 0
    min_loss = 100000
    min_index = len(sgd_losses) - 1
    for loss in sgd_losses:
        if loss < min_loss:
            min_loss = loss
            min_index = index
        index += 1
#         print(loss)

    return min_index, min_loss


# Stochastic gradient descent
def test_SGD(y, tX, ratio=0.2, w_initial=None, gamma=0.01, max_iters=1000, batch_size=1, seed=1):
    y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

    features = x_train.shape[1]
    
    if w_initial == None:
        w_initial = np.zeros(features)

    losses, w = least_squares_SGD(y_train, x_train, w_initial, batch_size, gamma, max_iters ,print_=False)
    
    min_index, _ = get_min_param_index(losses)

    s = score(x_test, y_test, w[min_index])
    
    return s
