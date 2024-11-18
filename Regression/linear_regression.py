from base import BaseRegression
import numpy as np


class LinearRegression(BaseRegression):
    """
    Linear Regression model.

    Uses Normal Equation to find optimal model parameters
    """
    def fit(self, X, y):
        """
        Train the linear model using NE

        Params:
        -------------
        X: numpy.ndarray
            Input features of matrix of shape(m, n)
        y: numpy.ndarray
            Target values of shape(m,)
        """

        # Add intercept term.
        X = np.c_[np.ones(X.shape[0], 1), X]

        # Calculate optimal parameters using NE algorithm
        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y


class RidgeRegression(BaseRegression):
    """
    Ridge regression model.

    Add L2 regularization term to prevent overfitting.

    Attr:
        alpha: float
            Regularization strength (L2 regularizer)
    """
    def __init__(self, alpha=1.0):
        """
        Initialize Ridge Regression with regularization parameter alpha.
        
        Parameters:
        ----------
        alpha : float, optional
            Regularization strength (default is 1.0).
        """
        super().__init__()
        self.alpha = alpha

    def fit():
        pass
