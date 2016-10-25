import numpy as np
from split_data import split_data
from proj1_helpers import predict_labels
from least_squares import least_squares
from gradient_descent import least_squares_GD
from stochastic_gradient_descent import least_squares_SGD
from ridge_regression import ridge_regression

def score(tX, y, w):
	labels = predict_labels(w, tX)
	return (labels == y).sum()/labels.size


def test_GD(y,tX,ratio,max_iters=1000,gamma=0.01,w_initial=None):
	seed = 1
	tX_tr, y_tr, tX_te, y_te = split_data(y, tX, ratio, seed)

	# Gradient descent
	if w_initial == None:
		w_initial = np.zeros(tX_tr.shape[1])
	_, w = least_squares_GD(y_tr, tX_tr, w_initial, max_iters, gamma)
	s = score(tX_te, y_te, w)
	print(s)

def test_SGD(y,tX,ratio,max_iters,gamma,batch_size,w_initial=None):
	seed = 1
	tX_tr, y_tr, tX_te, y_te = split_data(y, tX, ratio, seed)

	# Stochastic gradient descent
	if w_initial == None:
		w_initial = np.zeros(tX_tr.shape[1])
	_, w = least_squares_SGD(y_tr, tX_tr, w_initial, batch_size, max_iters, gamma)
	s = score(tX_te, y_te, w)
	print(s)

def test_LS(y,tX,ratio):
	seed = 1
	tX_tr, y_tr, tX_te, y_te = split_data(y, tX, ratio, seed)

	# Least squares
	w, _ = least_squares(y_tr, tX_tr)
	s = score(tX_te, y_te, w)
	print(s)

def test_RR(y, tX, ratio):
	seed = 1
	tX_tr, y_tr, tX_te, y_te = split_data(y, tX, ratio, seed)
	# Ridge regression
	w = ridge_regression(y_tr, tX_tr, 0.01)
	s = score(tX_te, y_te, w)
	print(s)

def test_LR(y, tX, ratio):
	seed = 1
	tX_tr, y_tr, tX_te, y_te = split_data(y, tX, ratio, seed)
	# Logistic regression
	print('not implemented')

def test_RLR(y, tX, ratio):
	seed = 1
	tX_tr, y_tr, tX_te, y_te = split_data(y, tX, ratio, seed)
	# Regularized logistic regression
	print('not implemented')