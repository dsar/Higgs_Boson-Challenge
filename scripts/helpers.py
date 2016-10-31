# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import csv
from costs import sigmoid

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

def build_poly(x, degree):
    """Build the polynomial expansion of x to the given degree"""
    N = x.shape[0]
    φ = np.ones([N, degree+1])
    for d in range(1,degree+1):
        φ[:,d] = x ** d
    return φ


def split_data(y, tx, a_indices):
    """Splits the data in two sets"""
    N = y.shape[0]
    b_indices = np.ones(N, dtype=np.bool)
    b_indices[a_indices] = 0

    y_a = y[a_indices]
    tx_a = tx[a_indices]    
    y_b = y[b_indices]
    tx_b = tx[b_indices]

    return y_a, tx_a, y_b, tx_b

def sample_data(y, x, seed, size_samples):
    """Sample points from the dataset"""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def predict_labels(weights, data, threshold=0):
    """Generates class predictions given weights, and a test data matrix"""

    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    return y_pred


def predict_logistic_labels(weights, data, threshold=0.5):
    """Generates class predictions given weights, and a test data matrix"""

    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    return y_pred

def score(y, tx, w, threshold=0):
    """calculates the final score of a given method"""
    labels = predict_labels(w, tx, threshold)
    count = 0
    for i in range(len(labels)):
        if labels[i] == y[i]:
            count += 1

    return count/len(labels)

def logistic_score(y, tx, w, threshold=0.5):
    """Generates class predictions given weights, and a test data matrix
     for logistic regression methods"""
    labels = predict_logistic_labels(w, tx, threshold)
    count = 0
    for i in range(len(labels)):
        if labels[i] == y[i]:
            count += 1

    return count/len(labels)