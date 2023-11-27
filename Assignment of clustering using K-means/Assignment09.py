import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the dataset
data = pd.read_csv('/content/iris.csv')  # Replace 'your_dataset.csv' with the actual path to your CSV file

# Extract features for clustering
features = data[['sepal_length', 'sepal_width']]

# Specify the number of clusters (change as needed)
k = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(features)

# Visualize the clusters
plt.figure(figsize=(10, 6))

# Define colors for each cluster
colors = ['blue', 'green', 'purple']

for cluster in range(k):
    cluster_data = data[data['cluster'] == cluster]
    plt.scatter(cluster_data['sepal_length'], cluster_data['sepal_width'], label=f'Cluster {cluster + 1}', color=colors[cluster])

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, color='red', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('kmeans_age_income_dataset.csv')

# Extracting features for clustering
X = data[['Age', 'Income']]

# Applying KMeans with k=4
kmeans = KMeans(n_clusters=4, random_state=42)  # Set n_clusters to 4
kmeans.fit(X)

# Adding cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Plotting the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data['Age'], data['Income'], c=data['Cluster'], cmap='viridis', edgecolors='k')
plt.title('K-means Clustering (k=4)')  # Update the title
plt.xlabel('Age')
plt.ylabel('Income')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

sse = kmeans.inertia_
print(f"Sum of Squared Errors (SSE): {sse}")
