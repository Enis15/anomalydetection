import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dask.array import indices

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

clusters = {
    'Dataset 1': {
        'DBSCAN' : {'eps': 0.45673548737193675, 'min_samples': 11},
        'LOF' : {'n_neighbors': 41}
    },
    'Dataset 2': {
        'DBSCAN': {'eps': 0.70215, 'min_samples': 31},
        'LOF': {'n_neighbors': 100}
    },
    'Dataset 3': {
        'DBSCAN': {'eps': 0.416300721723287, 'min_samples': 17},
        'LOF': {'n_neighbors': 100}
    },
    'Dataset 4': {
        'DBSCAN': {'eps': 0.6805623250067976, 'min_samples': 85},
        'LOF': {'n_neighbors': 100}
    }
}

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv')

# Dropping irrelevant columns for the anomaly detection
df = df.drop(['timestamp', 'sending_address', 'receiving_address'], axis=1)

# Relabeling column target column 'anomaly', where low risk:0, moderate & high risk =1
df['anomaly'] = df['anomaly'].replace({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})
df['anomaly'] = df['anomaly'].astype(int)

# Encoding categorical features with numerical variables
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

# Determining the X and y values
X = df.drop('anomaly', axis=1)
y = df['anomaly'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardize the data

# Define model and its parameters
eps = clusters['Dataset 4']['DBSCAN']['eps']
min_samples = clusters['Dataset 4']['DBSCAN']['min_samples']
n_neighbors = clusters['Dataset 4']['LOF']['n_neighbors']

def dbscan(X_scaled, eps, min_samples):
    clf = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clf.fit_predict(X_scaled)
    return labels

def lof(X_scaled, n_neighbors):
    clf = LocalOutlierFactor(n_neighbors=n_neighbors)
    labels = clf.fit_predict(X_scaled)
    return labels

labels = dbscan(X_scaled, eps, min_samples)

# Identify cluster points and anomalies
cluster_point = X_scaled[labels != -1]
anomalies = X_scaled[labels == -1]

# Compute the distance of the anomalies from the nearest cluster points
distances = []
nearest_points = []
for anomaly in anomalies:
    dists = euclidean_distances([anomaly], cluster_point)
    min_dits = np.min(dists)
    nearest_point = cluster_point[np.argmin(dists)]
    distances.append(min_dits)
    nearest_points.append(nearest_point)

# distance_csv = pd.DataFrame(distances).to_csv('../Extra/distances.csv')
# nearestneigh_csv = pd.DataFrame(nearest_points).to_csv('../Extra/nearestpoint.csv')

dist_nearest_csv = pd.DataFrame({
    'anomaly_index': np.where(labels == -1)[0],
    'distance': distances,
    'nearest_point_x':[point[0] for point in nearest_points],
    'nearest_point_y': [point[1] for point in nearest_points],
}).to_csv('../Extra/dist_nearest.csv', index=False)

# Visualize the results
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
cluster_points_pca = X_pca[labels != -1]
anomalies_pca = X_pca[labels == -1]

plt.figure(figsize=(10, 8))
plt.scatter(cluster_points_pca[:, 0], cluster_points_pca[:, 1], c='blue', label='Cluster Points')
plt.scatter(anomalies_pca[:, 0], anomalies_pca[:, 1], c='red', label='Anomalies', marker='x', s=100)
for i, anomaly in enumerate(anomalies_pca):
    nearest_point_pca = pca.transform([nearest_points[i]])[0] # Transform nearest points to 2 dimensional
    plt.plot([anomaly[0], nearest_point_pca[0]], [anomaly[1], nearest_point_pca[1]], c='black', linestyle='--')
plt.title('DBSCAN clustering with Anomalies and Nearest Cluster Points')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid(True)
plt.show()
