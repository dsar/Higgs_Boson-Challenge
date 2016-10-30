# -*- coding: utf-8 -*-
import numpy as np
from costs import compute_loss, compute_logistic_loss, sigmoid
from helpers import batch_iter

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    # TODO: compute gradient and loss
    e = y - (np.dot(tx,w))
    N = y.shape[0]
    
    gradient = -(1/N) * np.dot(np.transpose(tx),e)
    
    return gradient
	
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
	w = initial_w

	for n_iter in range(max_iters):
		gradient = compute_gradient(y, tx, w)
		loss = compute_loss(y,tx,w)
		w = w - gamma * gradient

	return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
	w = initial_w
	min_loss = compute_loss(y, tx, w)
	min_w = w

	for n_iter, (batch_y, batch_tx) in zip(range(max_iters), batch_iter(y, tx, 1)):
		gradient = compute_gradient(y, tx, w)
		loss = compute_loss(y, tx, w)
		w = w - gamma * gradient

		if loss < min_loss:
			min_loss = loss
			min_w = w

	return min_w, min_loss

def least_squares(y, tx):
	w = np.linalg.solve(np.transpose(tx).dot(tx), np.transpose(tx).dot(y))
	loss = compute_loss(y, tx, w)

	return w, loss

def ridge_regression(y, tx, lambda_):
	N = tx.shape[0]
	D = tx.shape[1]

	ident = np.eye(D) * lambda_ * 2 * N
	w = np.linalg.solve(np.transpose(tx).dot(tx) + ident, np.transpose(tx).dot(y))

	loss = compute_loss(y, tx, w)

	return w, loss

def compute_logistic_gradient(y, tx, w):
	"""Compute the gradient for logistic regression"""
	return np.transpose(tx).dot(sigmoid(tx.dot(w)) - y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
	w = initial_w

	for n_iter in range(max_iters):
		gradient = compute_logistic_gradient(y, tx, w)
		loss = compute_logistic_loss(y, tx, w)
		w = w - gamma * gradient

	return w, loss

def regularizers(lambda_, w):
	gradient_reg = 2 * lambda_ * w
	loss_reg = np.asscalar(lambda_ * np.transpose(tx).dot(w))

	return gradient_reg, loss_reg

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
	w = initial_w

	for n_iter in range(max_iters):
		gradient = compute_logistic_gradient(y, tx, w)
		loss = compute_logistic_loss(y, tx, w)
		gradient_reg, loss_reg = regularizers(lambda_, w)

		gradient = gradient - gradient_reg
		loss = loss - loss_reg
		w = w - gamma * gradient

	return w, loss