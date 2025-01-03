import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

# Create synthetic data for clustering
data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# Visualize the data
plt.scatter(data[:, 0], data[:, 1], s=50, alpha=0.5)
plt.title("Generated Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')  # n_init='auto' ensures compatibility with scikit-learn >= 1.4.0
kmeans.fit(data)

# Retrieve cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualize the Clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
