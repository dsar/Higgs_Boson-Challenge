import numpy as np
from split_data import split_data
from proj1_helpers import *
from helpers import get_min_param_index
from least_squares import least_squares
from gradient_descent import least_squares_GD
from stochastic_gradient_descent import least_squares_SGD
from ridge_regression import ridge_regression
from logistic_regression import logistic_regression_gradient_descent

def score(tX, y, w):
	labels = predict_labels(w, tX)
	return (labels == y).sum()/labels.size


# Gradient descent
def test_GD(y, tX, ratio=0.5, w_initial=None, gamma=0.01, max_iters=1000, seed=1):
	print('GD test')
	y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

	features = x_train.shape[1]
	
	if w_initial == None:
		w_initial = np.zeros(features)

	_, w = least_squares_GD(y_train, x_train, w_initial, gamma, max_iters ,print_=False)

	s = score(x_test, y_test, w[-1])

	print('GD score: ', s)
	print()

# Stochastic gradient descent
def test_SGD(y, tX, ratio=0.5, w_initial=None, gamma=0.01, max_iters=1000, batch_size=50, seed=1):
	print('SGD test')
	y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

	features = x_train.shape[1]
	
	if w_initial == None:
		w_initial = np.zeros(features)

	losses, w = least_squares_SGD(y_train, x_train, w_initial, batch_size, gamma, max_iters ,print_=False)
	
	min_index, _ = get_min_param_index(losses)

	s = score(x_test, y_test, w[min_index])
	return s

	print('SGD score: ',s)
	print()

# Least squares
def test_LS(y,tX,ratio=0.5,seed=1):
	#print('LS test')
	y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

	w, _ = least_squares(y_train, x_train)
	s = score(x_test, y_test, w)
	return s
	print("LS score: ",s)
	print()

# Ridge regression
def test_RR(y, tX, ratio=0.5,lambda_=0.01,seed = 1):
	print('RR test')
	y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)
	w = ridge_regression(y_train, x_train, lambda_)
	s = score(x_test, y_test, w)
	return s
	print('Ridge Regression score: ',s)
	print()

# Logistic regression
def test_LR(y, tX, ratio=0.5, seed=1, print_=False):
	print('LR test')
	y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)

	loss, w = logistic_regression_gradient_descent(y, tX,print_=False)
	s = score(x_test,y_test,w)
	return s
	print('Logistic Regression score: ',s)
	print()

# Regularized logistic regression
def test_RLR(y, tX, ratio=0.5,seed = 1):
	y_test, y_train, x_test, x_train  = split_data(y, tX, ratio, seed)
	return 0
	print('not implemented yet')
	print()