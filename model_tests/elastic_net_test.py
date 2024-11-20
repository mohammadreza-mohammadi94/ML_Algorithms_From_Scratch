#---------------------------------------------------------------------#
#                     Code By Mohammadreza Mohammadi                  #
#      Github:   https://github.com/mohammadreza-mohammadi94          #
#      LinkedIn: https://www.linkedin.com/in/mohammadreza-mhmdi/      #
#---------------------------------------------------------------------#

from Regression.linear_regression import Elasticnet
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

# Generate synthetic regression dataset
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize elastic net model 
elastic_net = Elasticnet(
    learning_rate=0.01,
    lambda_1=1.0,
    lambda_2=0.5,
    n_iterations=1000)

# Train the model
elastic_net.fit(X_train, y_train)

# Prediction
y_pred = elastic_net.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

#===============================================================================#
#================== Compare With ElasticNet From Scikit-learn ==================#
print("="*50)
print("ElasticNet Model From Scikit-learn:")
model = ElasticNet()
model.fit(X_test, y_test)
y_pred_model = model.predict(X_test)
mse_2 = mean_squared_error(y_test, y_pred_model)
r2_2 = r2_score(y_test, y_pred_model)
print(f"Mean Squared Error: {mse_2:.4f}")
print(f"R-squared: {r2_2:.4f}")
