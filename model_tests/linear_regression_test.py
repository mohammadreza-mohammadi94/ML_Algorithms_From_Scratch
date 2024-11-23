#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

from Regression.linear_models import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Generate synthetic regression dataset
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

# # Split into train and test sets
X_train, X_test = X[:160], X[160:]
y_train, y_test = y[:160], y[160:]

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

# Evaluate the regression model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Linear Regression Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

#===============================================================================#
#================== Compare With Lasso From Scikit-learn ==================#
print("="*50)
print("Linear Model From Scikit-learn:")

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_model = model.predict(X_test)

mse_2 = mean_squared_error(y_test, y_pred_model)
mae_2 = mean_absolute_error(y_test, y_pred_model)
r2_2= r2_score(y_test, y_pred_model)

print(f"Mean Squared Error (MSE): {mse_2:.4f}")
print(f"Mean Absolute Error (MAE): {mae_2:.4f}")
print(f"R² Score: {r2_2:.4f}")
