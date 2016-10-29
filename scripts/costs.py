# -*- coding: utf-8 -*-
import numpy as np

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def compute_rmse(mse):
    """Compute rmse from mse."""
    return np.sqrt(2*mse)

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-t))

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)
    # return calculate_mae(e)

def compute_logistic_loss(y, tx, w):
    """Calculate the negative log likelihood"""
    N = y.shape[0]
    return (1/N) * np.sum(np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w))
