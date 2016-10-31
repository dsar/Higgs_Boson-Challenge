# -*- coding: utf-8 -*-
import numpy as np
from implementations import *
from costs import compute_rmse, compute_loss, compute_logistic_loss
from helpers import split_data, score, logistic_score

def build_k_indices(N, k, seed):
    """build k indices for k-fold."""
    np.random.seed(seed) #make random numbers predictable
    indices = np.random.permutation(N)    
    interval = int(N / k)
    
    k_indices = [indices[i * interval: (i + 1) * interval] for i in range(k)]
    return np.array(k_indices)

def run_method(y, tx, method, params):
	"""Run the wanted method and returns its value"""
	if method == "LS":
		return least_squares(y, tx)
	elif method == "GD":
		return least_squares_GD(y, tx, params["initial_w"], params["max_iters"], params["gamma"])
	elif method == "SGD":
		return least_squares_SGD(y, tx, params["initial_w"], params["max_iters"], params["gamma"])
	elif method == "RR":
		return ridge_regression(y, tx, params["lambda_"])
	elif method == "LR":
		return logistic_regression(y, tx, params["initial_w"], params["max_iters"], params["gamma"])
	elif method == "RLR":
		return reg_logistic_regression(y, tx, params["lambda_"], params["initial_w"], params["max_iters"], params["gamma"])

def cross_validation(y, tx, method, params, k=5, seed=1):
	"""Gives the mean RMSE for running the given method with the given parameters"""
	N = y.shape[0]
	indices = build_k_indices(N, k, seed)
	s = np.zeros(k)
	l = np.zeros(k)

	for i in range(k):
		y_test, tx_test, y_train, tx_train = split_data(y, tx, indices[i])

		w, _ = run_method(y_train, tx_train, method, params)
		if method in [ "LR", "RLR" ]:
			l[i] = compute_rmse(compute_logistic_loss(y_test, tx_test, w))
			s[i] = logistic_score(y_test, tx_test, w)
		else:
			l[i] = compute_rmse(compute_loss(y_test, tx_test, w))
			s[i] = score(y_test, tx_test, w)

	return np.mean(l), np.mean(s)

"""Initial w doesn't matter and max_iters is based on what our computer can handle, so they aren't variable"""
def test_LS(y, tx):
	return cross_validation(y, tx, "LS", {})

def test_GD(y, tx, gamma):
	return cross_validation(y, tx, "GD", {
		"initial_w": np.zeros(tx.shape[1]),
		"max_iters": 100,
		"gamma"    : gamma
		})

def test_SGD(y, tx, gamma):
	return cross_validation(y, tx, "SGD", {
		"initial_w": np.zeros(tx.shape[1]),
		"max_iters": 100,
		"gamma"    : gamma
		})

def test_RR(y, tx, lambda_):
	return cross_validation(y, tx, "RR", {
		"lambda_"  : lambda_
		})

def test_LR(y, tx, gamma):
	return cross_validation(y, tx, "LR", {
		"initial_w": np.zeros(tx.shape[1]),
		"max_iters": 100,
		"gamma"    : gamma
		})

def test_RLR(y, tx, lambda_, gamma):
	return cross_validation(y, tx, "RLR", {
		"initial_w": np.zeros(tx.shape[1]),
		"max_iters": 100,
		"gamma"    : gamma,
		"lambda_"  : lambda_
		})
