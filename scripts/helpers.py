# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import matplotlib.pyplot as plt

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

# my funcions

def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]

def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx

def standardize_outliers(x):
    N = x.shape[0]
    D = x.shape[1]
    print(N, D)
    mean_x = np.zeros(D)
    std_x = np.zeros(D)
    for i in range(D):
        col = x[:,i]
        mean_x[i] = np.mean(col[col!=-999])
        std_x[i] = np.std(col[col!=-999])
        col[col==-999] = mean_x[i]
        col = (col-mean_x[i])/std_x[i] if std_x[i] != 0 else (col-mean_x[i])
        x[:,i] = col
    print(mean_x.shape)
    print(std_x.shape)
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x

def count_outliers(tX,outlier):
    features = tX.shape[1]
    sample_size = tX.shape[0]
    outliers = np.zeros(features)
    for feature in range(features):
        for row in range(sample_size):
            if tX[row,feature] == outlier:
                outliers[feature] += 1
    return outliers

def plot_features_by_y(y,tX):
    features = tX.shape[1]
    for feature in range(features):
        print('feature: ',feature)
        plt.scatter(tX[:,feature], y)
        plt.show()

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