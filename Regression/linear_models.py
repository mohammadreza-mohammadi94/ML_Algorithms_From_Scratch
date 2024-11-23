#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

# Import Libraries
from Regression.base import BaseRegression
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

        # Add bias term to X once here
        X = self._add_bias(X)  # Add the bias term once
        y = y.reshape(-1, 1)  # Ensure y is a column vector

        # Initialize theta
        self.theta = np.zeros((X.shape[1], 1))  # Match X columns including bias
        m = len(y)

        for _ in range(self.n_iterations):
            # Predict without adding the bias term again
            y_pred = self.predict(X, add_bias=False)  # Pass X without re-adding bias
            residuals = y_pred - y
            gradient = (1 / m) * X.T.dot(residuals)
            self.theta -= self.learning_rate * gradient

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

    def fit(self, X, y):
        # Add intercept term.
        X = np.c_[np.ones(X.shape[0], 1), X]

        # Calculate optimal parameters using NE algorithm based on L2 regularizer
         # Identity matrix for regularization (exclude intercept from regularization)
        I = np.eye(X.shape[1])
        I[0, 0] = 0 # Do not regularize the intercept

        self.theta = np.linalg.inv(X.T @ X + self.alpha @ I) @ X.T @ y


class LassoRegression(BaseRegression):
    """
    Lasso Regression model with L1 regularization (Lasso).

    Parameters:
    ----------
    learning_rate : float
        The learning rate for gradient descent.
    lambda_ : float
        Regularization strength (lambda).
    n_iterations : int
        Number of iterations for gradient descent.
    """

    def __init__(self,
                 learning_rate=0.01,
                 lambda_=1.0,
                 n_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambda_ = lambda_
        self.n_iterations = n_iterations

    def _soft_threshold(self, z, lambda_):
        """
        Apply the soft-thresholding operator to encourage sparsity.
        
        Parameters:
        ----------
        z : float
            The value to threshold.
            z refers to the coefficient or weight of specific feature
            in the regression model that is being updated during optimization.
        lambda_ : float
            The regularization parameter.

        Returns:
        -------
        float
            Thresholded value.
        -------------------------------------------
        The _soft_threshold() method is an implementation of the soft-thresholding
        operator used in Lasso Regression (and other L1 regularized models).
        Its purpose is to encourage sparsity in the model's coefficients by
        shrinking some coefficients to zero during the training process.

        During the optimization of Lasso, the gradient descent updates the coefficients.
        The soft-thresholding function is used to modify the coefficients after 
        the gradient update. This ensures that some coefficients become zero,
        effectively selecting the most relevant features.
        Example:
        if the current coefficient is large, it get reduced by lambda_
        if the coefficient is small(near to zero), it is forced to zero.

        Conditions:
        - if z is greater than lambda_, the coefficient is reduced by lambda_
        - if z is less than -lambda_, the coefficient is increased by lambda_
        - if the abs value of z is less/euqal than to lambda_, the coefficient,
            is set to zero, which is the key to enforcing sparsity.
        
        
        """
        return np.sign(z) * np.maximum(np.abs(z) - lambda_, 0)

    def fit(self, X, y):
        """
        Fit the Lasso Regression model to the training data.
        
        Parameters:
        ----------
        X : numpy.ndarray
            Input feature matrix of shape (m, n).
        y : numpy.ndarray
            Target values of shape (m,).
        """

        m, n = X.shape
        X = np.c_[np.ones(m), X]  # Add intercept term
        self.theta = np.zeros(n + 1)  # Initialize weights and bias

        prev_theta = self.theta.copy()
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)  # Full matrix for prediction
            residual = y - y_pred

            # Update bias (intercept term)
            self.theta[0] -= self.learning_rate * (-2 / m) * np.sum(residual) 

            # Update weights
            gradients = (-2 / m) * X[:, 1:].T.dot(residual)  # Gradients for weights
            self.theta[1:] = self._soft_threshold(self.theta[1:] - self.learning_rate * gradients, 
                                                  self.lambda_ * self.learning_rate)

            # Check for convergence
            if np.all(np.abs(self.theta - prev_theta) < 1e-6):
                break
            prev_theta = self.theta.copy()

class Elasticnet(BaseRegression):
    """
    ElasticNet Regression model with combined L1 (Lasso) and L2 (Ridge) regularization.

    Parameters:
    ----------
    learning_rate : float
        The learning rate for gradient descent.
    lambda_1 : float
        Regularization strength for L1 penalty.
    lambda_2 : float
        Regularization strength for L2 penalty.
    n_iterations : int
        Number of iterations for gradient descent.
    """

    def __init__(self,
                 learning_rate=0.01, 
                 lambda_1=1.0,
                 lambda_2=1.0,
                 n_iterations=1000):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_iterations = n_iterations

    def _soft_threshold(self, z, lambda_):
        """
        Apply the soft-thresholding operator to encourage sparsity.
        
        Parameters:
        ----------
        z : float
            The value to threshold.
            z refers to the coefficient or weight of specific feature
            in the regression model that is being updated during optimization.
        lambda_ : float
            The regularization parameter.

        Returns:
        -------
        float
            Thresholded value.
        -------------------------------------------
        The _soft_threshold() method is an implementation of the soft-thresholding
        operator used in Lasso Regression (and other L1 regularized models).
        Its purpose is to encourage sparsity in the model's coefficients by
        shrinking some coefficients to zero during the training process.

        During the optimization of Lasso, the gradient descent updates the coefficients.
        The soft-thresholding function is used to modify the coefficients after 
        the gradient update. This ensures that some coefficients become zero,
        effectively selecting the most relevant features.
        Example:
        if the current coefficient is large, it get reduced by lambda_
        if the coefficient is small(near to zero), it is forced to zero.

        Conditions:
        - if z is greater than lambda_, the coefficient is reduced by lambda_
        - if z is less than -lambda_, the coefficient is increased by lambda_
        - if the abs value of z is less/euqal than to lambda_, the coefficient,
            is set to zero, which is the key to enforcing sparsity.
        """
        return np.sign(z) * np.maximum(np.abs(z) - lambda_, 0)

    def fit(self, X, y):
        """
        Fit the ElasticNet Regression model to the training data.

        Parameters:
        ----------
        X : numpy.ndarray
            Input feature matrix of shape (m, n).
        y : numpy.ndarray
            Target values of shape (m,).
        """

        m, n = X.shape # Rows, Cols
        X = np.c_[np.ones(m), X]
        self.theta = np.zeros(n + 1)

        prev_theta = self.theta.copy()
        for _ in range(self.n_iterations):
            y_pred = self.predict(X)
            residual = y - y_pred

            # Update bias 
            self.theta[0] -= self.learning_rate * (-2 / m) * np.sum(residual)

            # Update weights
            for j in range(1, n + 1):
                gradient = (-2 / m) * np.dot(X[:, j], residual) + 2 * self.lambda_2 * self.theta[j]
                self.theta[j] -= self.learning_rate * gradient
                self.theta[j] = self._soft_threshold(self.theta[j], self.lambda_1 * self.learning_rate)
            
            # Check for convergence
            if np.all(np.abs(self.theta - prev_theta) < 1e-6):
                break
            prev_theta = self.theta.copy()

class PolynomialRegression:
    """
    Polynomial Regression model using Gradient Descent.

    Parameters:
    ----------
    degree : int
        The degree of the polynomial to fit to the data.
    learning_rate : float
        The learning rate for gradient descent optimization.
    n_iterations : int
        The number of iterations for the gradient descent algorithm.
    
    Attributes:
    ----------
    theta : numpy.ndarray
        The model parameters (weights and bias).
    """
    def __init__(self,
                 degree,
                 learning_rate=0.01,
                 n_iterations=1000):
        """
        Initialize the Polynomial Regression model.
        
        Parameters:
        ----------
        degree : int, optional
            The degree of the polynomial. Default is 2.
        learning_rate : float, optional
            The learning rate for gradient descent. Default is 0.01.
        n_iterations : int, optional
            The number of iterations for gradient descent. Default is 1000.
        """
        super().__init__()
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def _add_poly_features(self, X):
        """
        Generate polynomial features up to the specified degree.

        Parameters:
        ----------
        X : numpy.ndarray
            Input feature matrix of shape (m, 1), where m is the number of samples.

        Returns:
        -------
        numpy.ndarray
            Transformed feature matrix with polynomial terms up to the given degree.
        """
        X_poly = X.copy()
        for power in range(2, self.degree + 1):
            X_poly = np.c_[X_poly, X**power]
        return X_poly

    def fit(self, X, y):
        """
        Fit the Polynomial Regression model to the training data using gradient descent.

        Parameters:
        ----------
        X : numpy.ndarray
            Input feature matrix of shape (m, 1), where m is the number of samples.
        y : numpy.ndarray
            Target values of shape (m, 1).
        """
        X_poly = self._add_poly_features(X)
        
        # Add Bias
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]
        
        # Initialize weights correctly
        self.theta = np.zeros(X_poly.shape[1])  # Match number of features
        
        m = len(y)  # Number of samples

        for _ in range(self.n_iterations):
            y_pred = X_poly.dot(self.theta)
            residuals = y_pred - y.ravel()

            # Compute the gradient
            gradients = (2 / m) * X_poly.T.dot(residuals)
            
            # Update weights
            self.theta -= self.learning_rate * gradients

    def predict(self, X):
        """
        Predict target values for a given input feature matrix.

        Parameters:
        ----------
        X : numpy.ndarray
            Input feature matrix of shape (m, 1).

        Returns:
        -------
        numpy.ndarray
            Predicted values of shape (m, 1).
        """
        X_poly = self._add_poly_features(X)
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]
        return X_poly.dot(self.theta)