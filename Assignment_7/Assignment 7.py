import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'customer_spending_scattered_data.csv'
data = pd.read_csv(file_path)

# Extract the relevant features (ignoring the 'Cluster' column)
X = data[['Spending on Food', 'Spending on Clothing']]

# Initialize K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(X)

# Predict the cluster labels
clusters = kmeans.predict(X)

# Add the new cluster labels to the dataset
data['New Cluster'] = clusters

# Plot the clusters with centroids
plt.figure(figsize=(8, 6))
plt.scatter(X['Spending on Food'], X['Spending on Clothing'], c=clusters, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title("K-Means Clustering of Customer Spending")
plt.xlabel("Spending on Food")
plt.ylabel("Spending on Clothing")
plt.legend()
plt.grid(True)
plt.show()

# Step 1: Display the cluster centers (centroids)
centroids = kmeans.cluster_centers_
print("Cluster Centers (Centroids):")
print(pd.DataFrame(centroids, columns=['Spending on Food', 'Spending on Clothing']))

# Step 2: Explore the distribution of customers in each cluster
cluster_distribution = data['New Cluster'].value_counts()
print("\nCustomer Distribution Across Clusters:")
print(cluster_distribution)

# Display all rows of the dataset with the new cluster labels
print(data[['Spending on Food', 'Spending on Clothing', 'New Cluster']].to_string())

