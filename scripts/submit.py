from proj1_helpers import *
from poly import *
from least_squares import *
from helpers import *

def create_submission():
	# Load training data
	y_tr, tX_tr, _ = load_csv_data("../Data/train.csv")
	tX_tr, _, _ = standardize_outliers(tX_tr)
	print("Training data loaded")
		
	# Compute weights
	best_deg = find_best_poly(y_tr, tX_tr[:,1:])
	opt_tr = build_optimal(tX_tr[:,1:], best_deg)
	w, _ = least_squares(y_tr, opt_tr)
	print(score(opt_tr, y_tr, w))
	print("Weights computed")

	# Load test data
	_, tX_te, ids = load_csv_data("../Data/test.csv")
	tX_te, _, _ = standardize_outliers(tX_te)
	opt_te = build_optimal(tX_te[:,1:], best_deg)
	print("Test data loaded")

	# Create output
	y_te = predict_labels(w, opt_te)
	create_csv_submission(ids, y_te, "../Data/results.csv")
	print("Labels predicted")

create_submission()