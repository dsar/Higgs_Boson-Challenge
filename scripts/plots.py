# -*- coding: utf-8 -*-
"""visualize the result."""
import numpy as np
import matplotlib.pyplot as plt

from helpers import de_standardize, standardize


def visualization(y, x, mean_x, std_x, w, save_name):
    """visualize the raw data as well as the classification result."""

    #~~~DEBUGGING~~~
    print('---visualization---START')
    print('x shape: ',x.shape)
    print('y shape:',y.shape)
    print('mean_x shape:',mean_x.shape)
    print('std_x shape:',std_x.shape)
    print('---visualization---END')


    fig = plt.figure()
    # plot raw data
    print('before standardize: x shape',x.shape)
    print('before standardize: mean x shape',mean_x.shape)
    print('before standardize: std x shape',std_x.shape)
    x = de_standardize(x, mean_x, std_x)
    print('after standardize: x shape',x.shape)
    print('after standardize: mean x shape',mean_x.shape)
    print('after standardize: std x shape',std_x.shape)
    ax1 = fig.add_subplot(1, 2, 1)
    males = np.where(y == 1)
    females = np.where(y == 0)
    ax1.scatter(
        x[males, 0], x[males, 1],
        marker='.', color=[0.06, 0.06, 1], s=20)
    ax1.scatter(
        x[females, 0], x[females, 1],
        marker='*', color=[1, 0.06, 0.06], s=20)
    ax1.set_xlabel("Height")
    ax1.set_ylabel("Weight")
    ax1.grid()
    # plot raw data with decision boundary
    ax2 = fig.add_subplot(1, 2, 2)
    height = np.arange(
        np.min(x[:, 0]), np.max(x[:, 0]) + 0.01, step=0.01)
    weight = np.arange(
        np.min(x[:, 1]), np.max(x[:, 1]) + 1, step=1)
    hx, hy = np.meshgrid(height, weight)
    hxy = (np.c_[hx.reshape(-1), hy.reshape(-1)] - mean_x) / std_x
    x_temp = np.c_[np.ones((hxy.shape[0], 1)), hxy]
    prediction = x_temp.dot(w) > 0.5
    prediction = prediction.reshape((weight.shape[0], height.shape[0]))
    ax2.contourf(hx, hy, prediction, 1)
    ax2.scatter(
        x[males, 0], x[males, 1],
        marker='.', color=[0.06, 0.06, 1], s=20)
    ax2.scatter(
        x[females, 0], x[females, 1],
        marker='*', color=[1, 0.06, 0.06], s=20)
    ax2.set_xlabel("Height")
    ax2.set_ylabel("Weight")
    ax2.set_xlim([min(x[:, 0]), max(x[:, 0])])
    ax2.set_ylim([min(x[:, 1]), max(x[:, 1])])
    plt.tight_layout()
    plt.savefig(save_name)