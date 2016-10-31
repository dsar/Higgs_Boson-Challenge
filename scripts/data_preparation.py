# -*- coding: utf-8 -*-
import numpy as np

def standardize_outliers(tx, undef):
    """Standardize the data and replace undefined values with the mean of the column"""
    N = tx.shape[0]
    D = tx.shape[1]

    mean = np.zeros(D)
    std = np.zeros(D)

    for i in range(D):
        col = tx[:,i]
        mean[i] = np.mean(col[col != undef])
        std[i] = np.std(col[col != undef])

        col[col == undef] = mean[i]
        if std[i] != 0:
            col = (col - mean[i]) / std[i]
        else:
            col = col - mean[i]
        tx[:,i] = col

    # add offset column
    #tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean, std

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

def de_standardize(tx, mean_x, std_x):
    """Reverse the procedure of standardization."""
    tx = tx * std_x
    tx = tx + mean_x
    return tx

def count_outliers(tX,outlier):
    """counts and prints the number of the outliers given as a param"""
    features = tX.shape[1]
    sample_size = tX.shape[0]
    outliers = np.zeros(features)
    for feature in range(features):
        for row in range(sample_size):
            if tX[row,feature] == outlier:
                outliers[feature] += 1
    return outliers

import matplotlib.pyplot as plt
def plot_features_by_y(y,tX):
    """plots all features with respect to y (one-by-one)"""
    features = tX.shape[1]
    for feature in range(features):
        print('feature: ',feature)
        plt.scatter(tX[:,feature], y)
        plt.show()