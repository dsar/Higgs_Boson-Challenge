import numpy as np
from math import floor
from proj1_helpers import predict_labels
from least_squares import least_squares
from gradient_descent import least_squares_GD
from stochastic_gradient_descent import least_squares_SGD
from ridge_regression import ridge_regression

def split_data(y, X, ratio, seed):
	np.random.seed(seed)
	combined = np.c_[X, y]
	np.random.shuffle(combined)
	split = floor(ratio * X.shape[0])
	X1 = combined[:split, :X.shape[1]]
	y1 = combined[:split, X.shape[1]:]
	X2 = combined[split:, :X.shape[1]]
	y2 = combined[split:, X.shape[1]:]
	return X1, y1, X2, y2

def score(tX, y, w):
	labels = predict_labels(w, tX)
	return (labels == y).sum()/labels.size

def test(y, tX, ratio):
	seed = 1
	tX_tr, y_tr, tX_te, y_te = split_data(y, tX, ratio, seed)

	# Least squares
	w, _ = least_squares(y_tr, tX_tr)
	s = score(tX_te, y_te, w)
	print(s)

	# Gradient descent
	"""
	max_iters = 1000
	gamma = 0.01
	w_initial = np.zeros(tX_tr.shape[1])
	_, w = gradient_descent(y_tr, tX_tr, w_initial, max_iters, gamma)
	s = score(tX_te, y_te, w)
	print(s)
	"""

	# Stochastic gradient descent
	"""
	max_iters = 1000
	gamma = 0.01
	batch_size = 50
	w_initial = np.zeros(tX_tr.shape[1])
	_, w = stochastic_gradient_descent(y_tr, tX_tr, w_initial, batch_size, max_iters, gamma)
	s = score(tX_te, y_te, w)
	print(s)
	"""

	# Ridge regression
	w = ridge_regression(y_tr, tX_tr, 0.01)
	s = score(tX_te, y_te, w)
	print(s)

	# Logistic regression

	# Regularized logistic regression
