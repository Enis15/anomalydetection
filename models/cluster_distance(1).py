import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from utils.eda import load_data, datasets, dataset1_preprocessing, dataset2_preprocessing, dataset3_preprocessing, \
    dataset4_preprocessing

clusters = {
    'dataset1': {
        'DBSCAN': {'eps': 0.45673548737193675, 'min_samples': 11},
    },
    'dataset2': {
        'DBSCAN': {'eps': 0.70215, 'min_samples': 31},
    },
    'dataset3': {
        'DBSCAN': {'eps': 0.416300721723287, 'min_samples': 17},
    },
    'dataset4': {
        'DBSCAN': {'eps': 0.6805623250067976, 'min_samples': 85},
    }
}
# Configuration
dataset_name = 'dataset2'  # Specify dataset
chunk_size = 100000  # Number of rows to process in each batch
file_path = datasets[dataset_name]

# DBSCAN parameters for the dataset
eps = clusters[dataset_name]['DBSCAN']['eps']
min_samples = clusters[dataset_name]['DBSCAN']['min_samples']

# Results storage
all_results = []

def process_chunk(df, dataset_name):
    # Preprocessing and feature selection
    df = dataset2_preprocessing(df, dataset_name)
    X = df.drop('isFraud', axis=1)
    y = df['isFraud'].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering with DBSCAN
    clf = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clf.fit_predict(X_scaled)

    # Calculate centroids and average distances for each cluster
    labels_unique = set(labels)
    centroids = {}
    cluster_distance = {}

    for label in labels_unique:
        if label != -1:
            cluster_points = X_scaled[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids[label] = centroid
            avg_distance = euclidean_distances(cluster_points, centroid.reshape(1, -1)).mean()
            cluster_distance[label] = {
                'centroid': centroid,
                'avg_distance': avg_distance
            }

    # Calculate distances for anomaly points
    anomaly_distances = {}
    anomaly_points = X_scaled[labels == -1]

    for i, anomaly in enumerate(anomaly_points):
        anomaly_distances[i] = {}
        for label, data in cluster_distance.items():
            centroid = data['centroid']
            distance_to_centroid = euclidean_distances(anomaly.reshape(1, -1), centroid.reshape(1, -1))[0][0]
            anomaly_distances[i][label] = distance_to_centroid

    # Compile results
    results = []
    for anomaly_i, distances in anomaly_distances.items():
        for cluster_label, distance_centroid in distances.items():
            avg_dist = cluster_distance[cluster_label]['avg_distance']
            results.append({
                'Anomaly Index': anomaly_i,
                'Cluster Index': cluster_label,
                'Distance to Centroid': distance_centroid,
                'Average Distance to Centroid': avg_dist
            })
    return results

# Process data in chunks and accumulate results
for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunk_results = process_chunk(chunk, dataset_name)
    all_results.extend(chunk_results)

# Save all results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv(f'../results/nearest_dist/{dataset_name}_cluster_batched.csv', index=False)

print("Batch processing completed. Results saved.")