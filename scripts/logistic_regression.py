# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1.0 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""

    #~~~DEBUGGING~~~
    # print('---calculate_loss---START')
    # print('tx shape: ',tx.shape)
    # print('w shape:',w.shape)
    # print('y shape:',y.shape)
    # print('---calculate_loss---END')

    # scalar = tx.dot(w)
    # print('scalar shape: ',scalar.shape)
    # log = np.log(1+np.exp(tx.dot(w)))
    # print('log: ',log.shape)
    # mul = np.transpose(y).dot(tx.dot(w))
    # print('(y * tx.dot(w)) shape: ',mul.shape)
    # final = log - mul
    # print('final shape: ',final.shape)

    # lab05 way
    # total_loss = np.sum(np.log(1+np.exp(tx.dot(w))) - y*tx.dot(w))

    # another way
    total_loss = np.sum(np.log(1+np.exp(tx.dot(w))) - np.transpose(y).dot(tx.dot(w)))

    return total_loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""

    #~~~DEBUGGING~~~
    # print('---calculate gradient---START')
    # print('tx shape: ',tx.shape)
    # print('w shape:',w.shape)
    # print('y shape:',y.shape)
    # print('---calculate gradient---END')

    # scalar = sigmoid(tx.dot(w))
    # print('scalar shape: ',scalar.shape)
    # sub = np.transpose(scalar) - y
    # print('sub shape: ',sub.shape)

    # lab05 way
    # gradient = np.transpose(tx).dot(sigmoid(tx.dot(w))-y)

    # another way
    gradient = np.transpose(tx).dot(np.transpose(np.transpose(sigmoid(tx.dot(w)))-y))
    
    return gradient

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient
    return loss, w