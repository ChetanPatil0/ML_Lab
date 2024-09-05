# without regressor function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('driving_dataset.csv')
print("Original Dataset:")
print(data)

# Prepare the data
x = data.iloc[:, 0].values.reshape(-1, 1)  # Independent variable
y = data.iloc[:, 1].values                # Dependent variable

print("\nIndependent variable (x):")
pr int(x)

print("\nDependent variable (y):")
print(y)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("\nTraining Data:")
print(pd.DataFrame({'Hours': x_train.flatten(), 'Risk Score': y_train}))

print("\nTest Data:")
print(pd.DataFrame({'Hours': x_test.flatten(), 'Risk Score': y_test}))

# Calculate parameters manually
n = len(x_train)
x_mean = np.mean(x_train)
y_mean = np.mean(y_train)

# Calculate slope (beta_1)
numerator = np.sum((x_train - x_mean) * (y_train - y_mean))
denominator = np.sum((x_train - x_mean) ** 2)
beta_1 = numerator / denominator

# Calculate intercept (beta_0)
beta_0 = y_mean - (beta_1 * x_mean)

# Predict values
y_pred = beta_0 + beta_1 * x_test

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.3f}")

# Print coefficients
print(f"\nRegression Coefficient (Slope): {beta_1:.3f}")
print(f"Intercept: {beta_0:.3f}")

# Display predictions
print("\nPredictions on Test Data:")
predictions_df = pd.DataFrame({'Hours': x_test.flatten(), 'Actual Risk Score': y_test, 'Predicted Risk Score': y_pred})
print(predictions_df)

# Plot results
plt.figure(figsize=(12, 6))

# Training data
plt.scatter(x_train, y_train, color='purple', label='Training data')

# Regression line for training data
plt.plot(x_train, beta_0 + beta_1 * x_train, color='cyan', label='Regression Line (Training Set)')

# Test data
plt.scatter(x_test, y_test, color='orange', label='Test data')

# Regression line for test data
plt.plot(x_test, y_pred, color='magenta', linestyle='--', label='Regression Line (Test Set)')

plt.title('Training and Test Set Results')
plt.xlabel('Hours Spent Driving')
plt.ylabel('Risk Score')
plt.legend()
plt.grid(True)
plt.show()
