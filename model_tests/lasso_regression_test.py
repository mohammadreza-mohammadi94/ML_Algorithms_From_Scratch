#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

from Regression.linear_regression import LassoRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Generate synthetic regression dataset
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create lasso regression instance from LassRegression class
# that is imported from linear_regression module with 
# its default parameters.
my_lasso = LassoRegression(learning_rate=0.01,
                           lambda_=0.1,
                           n_iterations=1000)
# Train lasso regression.
my_lasso.fit(X_train, y_train)
# Make predictions
y_pred = my_lasso.predict(X_test)

# Evaluate the regression model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Lasso Regression Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")


#===============================================================================#
#================== Compare With Lasso From Scikit-learn ==================#
print("="*50)
print("Lasso Model From Scikit-learn:")

from sklearn.linear_model import Lasso

model = Lasso()
model.fit(X_train, y_train)
y_pred_model = model.predict(X_test)

mse_2 = mean_squared_error(y_test, y_pred)
mae_2 = mean_absolute_error(y_test, y_pred)
r2_2= r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse_2:.4f}")
print(f"Mean Absolute Error (MAE): {mae_2:.4f}")
print(f"R² Score: {r2_2:.4f}")