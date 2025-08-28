#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Regression.linear_regression import MyLinearRegression

# 1. Generate Synthetic Data for demonstration
np.random.seed(42) # For reproducibility
n_samples = 100
n_features = 1 # We'll start with one feature for easy plotting

# Create a true underlying relationship: y = 4 + 3*x + noise
X_true = 2 * np.random.rand(n_samples, n_features) # Feature values between 0 and 2
y_true = 4 + 3 * X_true.flatten() + np.random.randn(n_samples) # Target values with some noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_true, y_true, test_size=0.2, random_state=42)


# 2. Use our custom MyLinearRegression implementation
my_model = MyLinearRegression()
my_model.fit(X_train, y_train) # Train our model

my_y_pred = my_model.predict(X_test) # Make predictions on the test set

print("--- MyLinearRegression Results ---")
print(f"Intercept: {my_model.intercept_:.4f}")
print(f"Coefficients: {my_model.weights_}")
print(f"Mean Squared Error (My): {mean_squared_error(y_test, my_y_pred):.4f}")
print(f"R-squared (My): {r2_score(y_test, my_y_pred):.4f}")


# 3. Use scikit-learn's LinearRegression for comparison
sklearn_model = LinearRegression()
sklearn_model.fit(X_train, y_train) # Train the sklearn model

sklearn_y_pred = sklearn_model.predict(X_test) # Make predictions

print("\n--- Scikit-learn LinearRegression Results ---")
print(f"Intercept: {sklearn_model.intercept_:.4f}")
print(f"Coefficients: {sklearn_model.coef_}")
print(f"Mean Squared Error (Sklearn): {mean_squared_error(y_test, sklearn_y_pred):.4f}")
print(f"R-squared (Sklearn): {r2_score(y_test, sklearn_y_pred):.4f}")


# 4. Plotting the results (for the single-feature case)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual Test Data', alpha=0.7) # Plot actual test data points
plt.plot(X_test, my_y_pred, color='red', linestyle='--', linewidth=2, label='MyLinearRegression Prediction') # Plot our model's predictions
plt.plot(X_test, sklearn_y_pred, color='green', linestyle=':', linewidth=2, label='Sklearn LinearRegression Prediction') # Plot sklearn's predictions

plt.title('Linear Regression Comparison (1 Feature)')
plt.xlabel('Feature Value')
plt.ylabel('Target Value')
plt.legend()
plt.grid(True)
plt.show()