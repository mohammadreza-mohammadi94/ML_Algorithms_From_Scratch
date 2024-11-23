#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

import numpy as np
from Regression.linear_models import PolynomialRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 + 2 * X + X**2 + np.random.rand(100, 1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Polynomial Regression model
model = PolynomialRegression(degree=2,
                             learning_rate=0.01,
                             n_iterations=5000)
model.fit(X_train, y_train)

# Prediction 
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")
#========================================================================================#
#================== Compare With Polynomial Features From Scikit-learn ==================#
print("="*50)
print("Polynomial Model From Scikit-learn:")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Transform the features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
 
# Train the model on the training set
lin = LinearRegression()
lin.fit(X_train_poly, y_train)

# Predict on the test set
y_pred_poly = lin.predict(X_test_poly)

# Evaluate the model
mse_2 = mean_squared_error(y_test, y_pred_poly)
r2_2 = r2_score(y_test, y_pred_poly)

print(f"Mean Squared Error (MSE): {mse_2:.4f}")
print(f"R² Score: {r2_2:.4f}")