import numpy as np
from helpers import *
from data_preparation import standardize_outliers
from feature_selection import best_feature_degrees, build_poly_by_feature
from implementations import least_squares, ridge_regression

# Load the training data
DATA_TRAIN_PATH = "../Data/train.csv"
y_train, tx_train, _ = load_csv_data(DATA_TRAIN_PATH)
print("Loaded training data with dimensions ", tx_train.shape)

# Standardize the data and replace undefined values with the mean, column by column
tx_train, _, _ = standardize_outliers(tx_train, -999)
# Get optimal maximal degree for each feature for polynomial expansion
best_degrees = best_feature_degrees(y_train, tx_train, least_squares, max_degree=12)
# Build the dataset according to the degrees
tx_train = build_poly_by_feature(tx_train, best_degrees)
print("Created expanded data with shape ", tx_train.shape)

# Compute weight vector
weights, _ = ridge_regression(y_train, tx_train, 1e-06)

# Load and standardize test set
DATA_TEST_PATH = "../Data/test.csv"
_, tx_test, ids_test = load_csv_data(DATA_TEST_PATH)
tx_test, _, _ = standardize_outliers(tx_test, -999)
# Build test data with the same shape as the training data
tx_test = build_poly_by_feature(tx_test, best_degrees)

# Generate prediction vector
y_pred = predict_labels(weights, tx_test)

# Save predictions to output file
OUTPUT_PATH = "../Data/results.csv"
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
