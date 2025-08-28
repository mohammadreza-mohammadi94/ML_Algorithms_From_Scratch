#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

import numpy as np

class MyLinearRegression:
    """
    A simple implementation of Linear Regression using the Ordinary Least Squares (OLS) method,
    solved with Singular Value Decomposition (SVD) for numerical stability.
    """
    def __init__(self):
        """
        Initializes the MyLinearRegression model.
        Attributes to be set after fitting:
        - coef_: The array of feature weights (coefficients).
        - intercept_: The bias term (y-intercept).
        - feature_names_in_: Names of input features, if available (for sklearn-like compatibility).
        """
        self.coef_ = None          # Stores the combined vector [bias, w1, w2, ...] after fitting
        self.intercept_ = None     # Stores the bias term (b)
        self.weights_ = None       # Stores the actual feature weights (w1, w2, ...)
        self.feature_names_in_ = None # Stores feature names if passed, similar to sklearn

    def _add_intercept(self, X):
        """
        Adds a column of ones to the feature matrix X to account for the intercept (bias) term.
        This allows the intercept to be treated as a weight for a constant feature (value 1).

        Parameters:
        -----------
        X : np.array
            The input feature matrix.

        Returns:
        --------
        np.array
            The feature matrix with an added column of ones for the intercept.
        """
        # Create a column vector of ones with the same number of rows as X
        ones_column = np.ones((X.shape[0], 1))
        # Horizontally stack the ones_column with the original X
        return np.hstack([ones_column, X])

    def fit(self, X, y):
        """
        Fits the linear regression model to the training data (X, y) using OLS.
        It solves the system Xw = y for w (the parameters including bias and weights)
        by calculating the pseudo-inverse of X via Singular Value Decomposition (SVD).

        Parameters:
        -----------
        X : np.array
            The input feature matrix. Each row represents a sample, each column a feature.
            Expected shape: (n_samples, n_features).
        y : np.array
            The target values (output). Expected shape: (n_samples,) or (n_samples, 1).

        Returns:
        --------
        self : object
            Returns the instance itself, allowing for method chaining.
        """
        # Ensure y is a 1D array for consistent calculations
        y = y.flatten() if y.ndim > 1 else y

        # Store feature names if X is a pandas DataFrame, for sklearn-like compatibility
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f'x{i}' for i in range(X.shape[1])]

        # 1. Prepare the feature matrix by adding a column of ones for the intercept.
        # This augmented matrix is used to find both weights and bias in one go.
        # X_intercept will have shape (n_samples, n_features + 1).
        X_intercept = self._add_intercept(X)

        # 2. Solve the Ordinary Least Squares (OLS) problem using SVD.
        # The equation to solve is X_intercept @ w = y.
        # SVD decomposes X_intercept into U * Sigma * V.T.
        # np.linalg.svd returns U, s (singular values), Vt (V transpose).
        # full_matrices=False is used for a "thin" SVD, which is more efficient
        # for matrices where number of rows != number of columns.
        U, s, Vt = np.linalg.svd(X_intercept, full_matrices=False)

        # 3. Compute the pseudo-inverse of X_intercept using the SVD components.
        # The pseudo-inverse X_plus = V @ Sigma_plus @ U.T.
        # First, create a diagonal matrix of inverse singular values (Sigma_plus).
        s_inv = np.zeros(s.shape)
        # We only invert non-zero singular values. A small threshold (1e-10) is used
        # to prevent division by very small numbers, which can lead to numerical instability.
        s_inv[s > 1e-10] = 1 / s[s > 1e-10]

        # Form the diagonal matrix Sigma_plus from the inverse singular values.
        Sigma_plus = np.diag(s_inv)

        # Calculate the pseudo-inverse of X_intercept.
        X_pseudo_inverse = Vt.T @ Sigma_plus @ U.T

        # 4. Calculate the parameter vector (w) using the pseudo-inverse.
        # w = X_pseudo_inverse @ y
        # This vector 'self.coef_' will contain [bias, w1, w2, ..., wn].
        self.coef_ = X_pseudo_inverse @ y

        # 5. Separate the bias term and feature weights.
        # The first element of 'self.coef_' is the bias.
        self.intercept_ = self.coef_[0]
        # The remaining elements are the weights for the features.
        self.weights_ = self.coef_[1:]

        return self # Return self for consistency with sklearn's API

    def predict(self, X):
        """
        Predicts target values for new data using the fitted linear model.

        Parameters:
        -----------
        X : np.array
            The input feature matrix for which to make predictions.
            Expected shape: (n_samples, n_features).

        Returns:
        --------
        np.array
            The predicted target values. Shape: (n_samples,).
        """
        # The prediction formula is Y_pred = X @ weights + intercept.
        # X is multiplied by the learned feature weights, and then the intercept is added.
        return X @ self.weights_ + self.intercept_

    def get_params(self, deep=True):
        """
        Returns parameters for the estimator. Used for sklearn compatibility.
        """
        return {}

    def set_params(self, **params):
        """
        Sets the parameters of the estimator. Used for sklearn compatibility.
        This simple implementation does not accept parameters in set_params.
        """
        if not params:
            return self
        raise ValueError("MyLinearRegression does not accept parameters in set_params.")