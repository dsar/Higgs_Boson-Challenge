### Notebook
- Section 1: preparation of the data (standardization and polynomial expansion)
- Section 2: find the best parameters through cross-validation
- Section 3: apply the best weights to the test data and output submission file

### Important files:
- implementations.py
	Contains implementations of the basic functions (no cross-validation or any fancy stuff)
- data_preparation.py
	Contains standardization code and data analysis functions
- feature_selection.py
	Previously build_polynomial.py, create the expanded polynomial from the original data
- cross_validation.py
	Implements cross-validation (test_** functions) to find best parameters (see notebook)

### Helpers
- helpers.py
- costs.py

### Notes
- Cross validation can take a lot of time. You can change the number of iterations in the test_** functions in cross-validation.py
