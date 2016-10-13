from costs import compute_loss
import numpy as np


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    loss = np.zeros((len(w0), len(w1)))
    # TODO: compute loss for each combination of w0 and w1.
    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            loss[i,j] = compute_loss(y, tx, np.array([w0[i], w1[j]]))
            
    return loss