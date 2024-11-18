import numpy as np

class BaseRegression:
    """
    Base class for regression models.
    
    Attributes:
    ----------
    theta : numpy.ndarray
        Model parameters (weights and bias).
    """
    def __init__(self):
        """
        Initialize the Regression model.
        """
        self.theta = None # Model's coefficients

    # Prediction method
    def predict(self, X):
        """
        Predict the target values for input features X.

        params:
        --------------------------------
        X: numpy.nbarray
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

        if self.theta is None:
            raise ValueError("Model has not been fitted yet.")
        
        if X.shape[1] + 1 == self.theta.shape[0]:
            X = np.c_[np.ones(X.shape[0], 1), X]
        
        # Calculate the dot product of the X and theta to produce the
        # target values
        return X.dot(self.theta) # np.dot(X, self.theta)
