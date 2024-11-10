import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('driving_backache_risk_dataset.csv')

# Display the first 10 rows
print(data.head(10))

# Splitting model data with 70% for training
X = data[['hours_spent_driving']]  # Features (hours spent driving)
Y = data['backache_risk']          # Target variable (backache risk)

# Split the data into training and testing sets (70% training, 30% testing)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=5)

# Using Linear Regression Model
lr = LinearRegression()

# Train the model on training data
lr.fit(x_train, y_train)

# Predict the testing data to evaluate the model
pred_lr = lr.predict(x_test)

# Model Evaluation: error for linear regression
mse_lr = mean_squared_error(y_test, pred_lr, squared=False)
print("Error for Linear Regression = {}".format(mse_lr))

# Example: Predicting backache risk for someone who drives 8 hours a day
hours_spent = pd.DataFrame({'hours_spent_driving': [8]})  # Match the feature name with the training data

# Use the model to make the prediction
predicted_risk = lr.predict(hours_spent)
print(f"Predicted backache risk for 8 hours of driving: {predicted_risk[0]}")

# Extract and print the coefficient (slope) of the model
coefficient = lr.coef_[0]
print(f"Coefficient (slope) for hours spent driving: {coefficient}")
