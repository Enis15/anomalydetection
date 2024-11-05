import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from utils.eda import load_data, datasets, dataset1_preprocessing, dataset2_preprocessing, dataset3_preprocessing, \
    dataset4_preprocessing

# Predefined parameters from hyperparameter tuning for DBSCAN
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

def X_y(df, dataset_name):
    """"
    Preprocess the dataset based on its name

    Parameters:
        df: the input dataframe (dataset)
        dataset_name (str): the name of the dataset, used to determine the preprocessing

    Returns:
        X: Feature matrix
        y: Target label
    """
    if dataset_name == 'dataset1':
        df = dataset1_preprocessing(df, dataset_name)
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud'].values
    if dataset_name == 'dataset2':
        df = dataset2_preprocessing(df, dataset_name)
        X = df.drop('isFraud', axis=1)
        y = df['isFraud'].values
    if dataset_name == 'dataset3':
        df = dataset3_preprocessing(df, dataset_name)
        X = df.drop('fraud', axis=1)
        y = df['fraud'].values
    if dataset_name == 'dataset4':
        df = dataset4_preprocessing(df, dataset_name)
        X = df.drop('anomaly', axis=1)
        y = df['anomaly'].values
    return X, y


if __name__ == '__main__':

    # Load the dataset
    dataset_name = 'dataset4'  # Change for the desired dataset
    file_path = datasets[dataset_name]
    df = load_data(file_path)

    # Determining the X and y values
    X, y = X_y(df, dataset_name)

    # Standardize the feature data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define model and its parameters from the 'clusters' dictionary
    model_name = 'DBSCAN'
    eps = clusters[dataset_name]['DBSCAN']['eps']
    min_samples = clusters[dataset_name]['DBSCAN']['min_samples']

    clf = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clf.fit_predict(X_scaled)

    # Append cluster labels to the dataset for reference
    df['cluster'] = labels

    # Calculate the centroid for each cluster
    labels_unique = set(labels) # Get only one cluster label without duplicates
    centroids = {}
    for label in labels_unique:
        if label != -1: # ignore the noise points
            cluster_point = X_scaled[labels == label]
            centroid = cluster_point.mean(axis=0)
            centroids[label] = centroid

    # Calculate average distance from each cluster point to its centroid
    cluster_distance = {}
    for label, centroid in centroids.items():
        cluster_point = X_scaled[labels == label]
        distance = euclidean_distances(cluster_point, centroid.reshape(1, -1)).flatten()
        avg_distance = distance.mean()
        cluster_distance[label] = {
            'centroid': centroid,
            'avg_distance': avg_distance,
            'distance': distance
        }
        #print(f'Cluster {label}: Average distance to centroid: {avg_distance}')

    # Calculate distance of each anomaly point to each cluster's centroid
    anomaly_distances = {}
    anomaly_points = X_scaled[labels == -1] # Points labeled as -1 are considered anomalies

    for i, anomaly in enumerate(anomaly_points):
        anomaly_distances[i] = {}
        for label, data in cluster_distance.items():
            centroid = data['centroid']
            distance_centroid = euclidean_distances(centroid.reshape(1, -1), anomaly.reshape(1, -1))[0][0]
            anomaly_distances[i][label] = distance_centroid
            #print(f'Anomaly{i}: Distance to cluster {label} centroid: {distance_centroid}')

    # Summarize the results
    results = []

    print('\n Summary of calculations:')
    for anomaly_i, distances in anomaly_distances.items():
        print(f'Anomaly {anomaly_i}')
        for cluster_label, distance_centroid in distances.items():
            avg_dist = cluster_distance[cluster_label]['avg_distance']
            print(f'Cluster {cluster_label}: Distance to Centroid = {distance_centroid},'
                  f'Average distance to centroid: {avg_dist}')
            results.append({
                'Anomaly Index': anomaly_i,
                'Cluster Index': cluster_label,
                'Distance to Centroid': distance_centroid,
                'Average Distance to Centroid': avg_dist
            })
    # Save results in a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'../results/nearest_dist/{dataset_name}_cluster.csv', index=False)

    # Aggregate the results by cluster, calculating statistics for anomaly distances for each centroid
    summary_results = results_df.groupby('Cluster Index').agg(
        avg_anomaly_distance=('Distance to Centroid', 'mean'),
        avg_centroid_dist=('Average Distance to Centroid', 'mean'),
        min_anomaly_distance=('Distance to Centroid', 'min'),
        max_anomaly_distance=('Distance to Centroid', 'max'),
    ).reset_index()

    # Calculate the ratio between average anomaly distance and average centroid distance for each cluster
    summary_results['ratio'] = (summary_results['avg_anomaly_distance'] / summary_results['avg_centroid_dist'])

    # Reorder the columns
    summary_results = summary_results[
        ['Cluster Index', 'avg_anomaly_distance', 'avg_centroid_dist', 'ratio', 'min_anomaly_distance',
         'max_anomaly_distance']
    ]
    # save summary results in a CSV file
    summary_csv = summary_results.to_csv(f'../results/nearest_dist/{dataset_name}_dist_summary.csv', index=False)
    print('Results saved in csv!')



