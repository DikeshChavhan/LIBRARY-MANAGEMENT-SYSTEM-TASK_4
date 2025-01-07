# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the dataset
iris = load_iris()
data = iris.data
feature_names = iris.feature_names

# Step 2: Preprocessing
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Step 3: Clustering Algorithms

# K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(data_scaled)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_scaled)

# Step 4: Evaluation

# Silhouette score
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
hierarchical_silhouette = silhouette_score(data_scaled, hierarchical_labels)
dbscan_silhouette = silhouette_score(data_scaled, dbscan_labels)

# Davies-Bouldin index
kmeans_db = davies_bouldin_score(data_scaled, kmeans_labels)
hierarchical_db = davies_bouldin_score(data_scaled, hierarchical_labels)
dbscan_db = davies_bouldin_score(data_scaled, dbscan_labels)

print("Silhouette Score:")
print(f"K-means: {kmeans_silhouette}")
print(f"Hierarchical: {hierarchical_silhouette}")
print(f"DBSCAN: {dbscan_silhouette}")

print("\nDavies-Bouldin Index:")
print(f"K-means: {kmeans_db}")
print(f"Hierarchical: {hierarchical_db}")
print(f"DBSCAN: {dbscan_db}")

# Step 5: Visualization

# K-means clustering visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('K-means Clustering')
plt.show()

# Hierarchical clustering visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=hierarchical_labels, palette='viridis')
plt.title('Hierarchical Clustering')
plt.show()

# DBSCAN clustering visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=dbscan_labels, palette='viridis')
plt.title('DBSCAN Clustering')
plt.show()
