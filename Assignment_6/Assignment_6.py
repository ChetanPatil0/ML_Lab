import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('student_performance.csv')
print(df)
# Define features and target
X = df[['Hours_Studied', 'Hours_Slept']]
y = df['Passed']

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors classifier with 60 neighbors
knn = KNeighborsClassifier(n_neighbors=60)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Plotting the decision boundary

# Create a mesh grid
h = 0.2  # step size in the mesh
x_min, x_max = X['Hours_Studied'].min() - 1, X['Hours_Studied'].max() + 1
y_min, y_max = X['Hours_Slept'].min() - 1, X['Hours_Slept'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Create a DataFrame for the mesh grid with the same feature names as the training data
grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)

# Predict on the mesh grid
Z = knn.predict(grid_points)
Z = Z.reshape(xx.shape)

# Plot the contours
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdYlBu)
plt.scatter(X_train['Hours_Studied'], X_train['Hours_Slept'], c=y_train, marker='o', edgecolor='k', label='Training data')
plt.scatter(X_test['Hours_Studied'], X_test['Hours_Slept'], c=y_test, marker='x', label='Test data')  # Removed edgecolor='k'

plt.xlabel('Hours Studied')
plt.ylabel('Hours Slept')
plt.title('KNN Classification Decision Boundary')
plt.legend()
plt.show()
