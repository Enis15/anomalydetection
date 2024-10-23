import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from utils.eda import load_data, datasets, dataset1_preprocessing, dataset2_preprocessing, dataset3_preprocessing, dataset4_preprocessing

clusters = {
    'dataset1': {
        'DBSCAN' : {'eps': 0.45673548737193675, 'min_samples': 11},
        'LOF' : {'n_neighbors': 41}
    },
    'dataset2': {
        'DBSCAN': {'eps': 0.70215, 'min_samples': 31},
        'LOF': {'n_neighbors': 100}
    },
    'dataset3': {
        'DBSCAN': {'eps': 0.416300721723287, 'min_samples': 17},
        'LOF': {'n_neighbors': 100}
    },
    'dataset4': {
        'DBSCAN': {'eps': 0.6805623250067976, 'min_samples': 85},
        'LOF': {'n_neighbors': 100}
    }
}

def cluster_model(X_scaled, model_name, eps=None, min_samples=None, n_neighbors=None):
    if model_name == 'DBSCAN':
        clf = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clf.fit_predict(X_scaled)
    elif model_name == 'LOF':
        clf = LocalOutlierFactor(n_neighbors=n_neighbors)
        labels = clf.fit_predict(X_scaled)
    else:
        print('Invalid model!')

    return labels

def X_y(df, dataset_name):
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

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # Standardize the data

    # Define model and its parameters
    model_name = 'DBSCAN' # Replace with LOF for
    eps = clusters[dataset_name]['DBSCAN']['eps']
    min_samples = clusters[dataset_name]['DBSCAN']['min_samples']
    n_neighbors = clusters[dataset_name]['LOF']['n_neighbors']

    # Apply the model
    labels = cluster_model(X_scaled, model_name, eps, min_samples, n_neighbors)

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

    # Save the results in a csv file
    dist_nearest_csv = pd.DataFrame({
        'anomaly_index': np.where(labels == -1)[0],
        'distance': distances,
        'nearest_point_x':[point[0] for point in nearest_points],
        'nearest_point_y': [point[1] for point in nearest_points],
    }).to_csv(f'../results/nearest_dist/{dataset_name}_{model_name}_neardist.csv', index=False)

    # Visualize the results
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    cluster_points_pca = X_pca[labels != -1]
    anomalies_pca = X_pca[labels == -1]

    # Select a random anomaly point
    random_anomaly = np.random.choice(len(anomalies_pca))

    # Visualize the anomaly point and the clusters
    plt.figure(figsize=(10, 8))
    plt.scatter(cluster_points_pca[:, 0], cluster_points_pca[:, 1], c='blue', label='Cluster Points')
    plt.scatter(anomalies_pca[random_anomaly, 0], anomalies_pca[random_anomaly, 1], c='red', label='Anomaly', marker='.', s=100)

    nearest_point_pca = pca.transform([nearest_points[random_anomaly]])[0]
    plt.plot([anomalies_pca[random_anomaly, 0], nearest_point_pca[0]], [anomalies_pca[random_anomaly, 1], nearest_point_pca[1]], c='black', linestyle='--')

    plt.title(f'{model_name} clustering with Anomalies and Nearest Cluster Points ({dataset_name})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.grid(True)
    plt.show()

    # for i, anomaly in enumerate(anomalies_pca):
    #     nearest_point_pca = pca.transform([nearest_points[i]])[0] # Transform nearest points to 2 dimensional
    #     plt.plot([anomaly[0], nearest_point_pca[0]], [anomaly[1], nearest_point_pca[1]], c='black', linestyle='--')