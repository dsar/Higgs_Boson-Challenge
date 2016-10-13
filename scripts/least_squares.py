import costs
import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    # least squares: TODO
    # returns mse, and optimal weights
    w_opt = np.linalg.inv(np.transpose(tx).dot(tx)).dot(np.transpose(tx)).dot(y)
    mse = costs.compute_loss(y, tx, w_opt)
    print('loss: ', mse)
    print('parameters w: ', w_opt)
    return w_opt, mse