#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

import numpy as np


class BaseRegression:
    """
    Base class for regression models.
    
    Attributes:
    ----------
    theta : numpy.ndarray
        Model parameters (weights and bias).
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Regression model.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None # Model's coefficients

    
    def _add_bias(self, X):
        """
        Adds a bias term (column of ones) to the input feature matrix.

        Parameters:
        ----------
        X : numpy.ndarray
            Input feature matrix.

        Returns:
        -------
        numpy.ndarray
            Feature matrix with bias term.
        """
        return np.c_[np.ones((X.shape[0], 1)), X]

    # Prediction method
    def predict(self, X, add_bias=True):
        """
        Predict the target values for input features X.

        params:
        --------------------------------
        X: numpy.ndarray
            Input feature matrix of shape (m,n)

        Returns:
        --------------------------------
        y_pred : numpy.ndarray
            Predicted target values of shape (m,)

        Raises:
        --------------------------------
        ValueError:
            if the model is not trained yet then theta is none.
        """
        # Calculate the dot product of X and theta to produce the target values
        if add_bias:
            X = self._add_bias(X)  # Add bias only if needed
        return X.dot(self.theta)  # Compute predictions# or np.dot(X, self.theta)
