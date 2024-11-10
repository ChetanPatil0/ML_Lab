import numpy as np
import pandas as pd
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("customer_spending_scattered_data.csv")

# Select the relevant features (Spending on Food and Spending on Clothing)
X = data[['Spending on Food', 'Spending on Clothing']]

# Optionally, standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define number of clusters
n_clusters = 5

# Implement K-Medoids clustering
kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
kmedoids.fit(X_scaled)

# Get cluster labels and medoids
labels = kmedoids.labels_
medoid_indices = kmedoids.medoid_indices_
medoids = X_scaled[medoid_indices]  # Get actual medoid values from scaled data

# Output results
print("Cluster labels:", labels)
print("Medoid indices:", medoid_indices)
print("Medoid values:", medoids)

# Visualize clusters with grid lines
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(medoids[:, 0], medoids[:, 1], c='red', marker='X', s=200)  # Medoids
plt.title('K-Medoids Clustering with Gridlines')
plt.xlabel('Spending on Food (Standardized)')
plt.ylabel('Spending on Clothing (Standardized)')
plt.grid(True)  # Add gridlines
plt.legend(['Data Points', 'Medoids'])
plt.show()
