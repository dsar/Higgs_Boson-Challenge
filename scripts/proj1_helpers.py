# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt


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


########my_helpers############

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
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    print(tx.shape)
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


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed) #make random numbers predictable
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def kfold_split_data(x, y,test_indices):
    
    N = y.shape[0]
    
    x_test = x[test_indices]
    y_test = y[test_indices]
    
    train_indices = [item for item in range(N) if item not in test_indices]
    x_train = x[train_indices]
    y_train = y[train_indices]
    
    return x_test,x_train,y_test,y_train

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""

    np.random.seed(seed)
    N = x.shape[0]
    
    indices = np.arange(N)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    
    return x[:int(N*ratio)], x[int(N*ratio):], y[:int(N*ratio)], y[int(N*ratio):]

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))

def score(tX, y, w, threshold=0):
    labels = predict_labels(w, tX,threshold)
    count = 0
    for i in range(len(labels)):
        if labels[i] == y[i]:
            count += 1

    return count/len(labels)

def logistic_score(tX, y, w, threshold=0.5):
    labels = predict_logistic_labels(w, tX,threshold)
    count = 0
    for i in range(len(labels)):
        if labels[i] == y[i]:
            count += 1

    return count/len(labels)