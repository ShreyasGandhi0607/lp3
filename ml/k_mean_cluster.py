import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("K-means-Cluster/sales_data_sample.csv",  encoding='latin-1')

print(data.head())

features = data[['SALES', 'QUANTITYORDERED', 'PRICEEACH']].dropna()

#normalize

scaler = StandardScaler()
scaled_feature = scaler.fit_transform(features)

#elbow method to find the optimal number of clusters

inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters = k, random_state = 0, n_init=10)
    kmeans.fit(scaled_feature)
    inertia.append(kmeans.inertia_)

#plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

#from the plot the optimal number of cluster is 3
optimal_k = 3

#apply k means with the optimal number of k

kmeans = KMeans(n_clusters = optimal_k, random_state = 0, n_init=10)
clusters = kmeans.fit_predict(scaled_feature)

#Add the clusters to the original dataste
features['Cluster'] = clusters

print(features.head())

#visualize the cluster
plt.figure(figsize = (8, 5))
plt.scatter(scaled_feature[:, 0], scaled_feature[:, 1], c = clusters, cmap = 'viridis')
plt.title(f'K-means with optimal number of clusters {optimal_k}')
plt.xlabel('Feature 1 : SALES')
plt.ylabel('Feature 2 : QUANTITYORDERED')
plt.show()