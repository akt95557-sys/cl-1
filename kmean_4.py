# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# 2. Load Iris dataset
iris = load_iris()
data = iris.data

# 3. Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 4. Find optimum clusters using Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        random_state=42,
        n_init=10
    )
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# 5. Plot Elbow graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()
wcss

