import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Import the models
from pyod.models.iforest import IForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from sklearn.cluster import KMeans, DBSCAN

# Define function for LOF (Local Outlier Factor) Algorithm
def model_lof(X, y, n_neighbors):
    """
    LOF Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        n_neighbors: number of neighbors.

    Returns:
        tuple: roc_auc score, f1 score and runtime of LOF algorithm.

    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', n_jobs=-1)

    # Get the prediction labels and scores
    X_labels = model.fit_predict(X)  # Outlier labels (1 = outliers & -1 = inliners)

    # Convert LOF labels (-1, 1) to (1, 0)
    X_labels = (X_labels == -1).astype(int)

    #Evaluation metrics
    roc_auc_lof = round(roc_auc_score(y, X_labels), 3)
    f1_score_lof = round(f1_score(y, X_labels, average='weighted'), 3)
    runtime_lof = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for LOF model, with n_neighbors = {n_neighbors}, are: \n'
          f'ROC AUC: {roc_auc_lof}\n'
          f'F1 score: {f1_score_lof}\n' 
          f'Time elapsed: {runtime_lof}')

    return roc_auc_lof, f1_score_lof, runtime_lof

#Define function for K-Mmeans Algorithm
def model_kmeans(X, y, k):
    """
    K-Means Clustering Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        k: number of clusters.

    Returns:
    tuple: roc_auc score, f1 score and runtime of K-Means Clustering.
  """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = KMeans(n_clusters=k, init='random', random_state=42)
    model.fit(X)

    #Get the prediction labels and scores
    X_labels = model.predict(X)  #Outlier labels (1 = outliers & 0 = inliers)

    #Evaluation metrics
    roc_auc_kmeans= round(roc_auc_score(y, X_labels), 3)
    f1_score_kmeans = round(f1_score(y, X_labels, average='weighted'), 3)
    runtime_kmeans = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for K-Means model, with k {k} are: \n'
          f'ROC AUC: {roc_auc_kmeans}\n'
          f'F1 score: {f1_score_kmeans}\n' 
          f'Time elapsed: {runtime_kmeans}')

    return roc_auc_kmeans, f1_score_kmeans, runtime_kmeans


# Define function for IForest (Isolation Forest) Algorithm
def model_iforest(X, y, n_estimators):
    """
    Isolation Forest Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        n_estimators: Number of trees in the forest.

    Returns:
        tuple: roc_auc score, f1 score and runtime of Isolation Forest algorithm.
  """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = IForest(n_estimators=n_estimators, random_state=42)
    model.fit(X)

    # Get the prediction labels and scores
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliners)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_iforest = round(roc_auc_score(y, X_labels), 3)
    f1_score_iforest = round(f1_score(y, X_labels, average='weighted'), 3)
    runtime_iforest= round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Isolation Forest model, with n_estimators = {n_estimators}, are: \n'
          f'ROC AUC: {roc_auc_iforest}\n'
          f'F1 score: {f1_score_iforest}\n'
          f'Time elapsed: {runtime_iforest}')

    return roc_auc_iforest, f1_score_iforest, runtime_iforest

# Define function for PCA (Principal Component Analysis) Algorithm
def model_pca(X, y):
    """
    PCA Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics


    Returns:
        tuple: roc_auc score, f1 score and runtime of PCA algorithm.
  """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = PCA()
    model.fit(X)

    # Get the prediction labels and scores
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliners)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_pca = round(roc_auc_score(y, X_labels), 3)
    f1_score_pca = round(f1_score(y, X_labels, average='weighted'), 3)
    runtime_pca = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for PCA model are: \n'
          f'ROC AUC: {roc_auc_pca}\n'
          f'F1 score: {f1_score_pca}\n'
          f'Time elapsed: {runtime_pca}')

    return roc_auc_pca, f1_score_pca, runtime_pca

# Define function for COPOD (Copula-Based Outlier Detection) Algorithm
def model_copod(X, y):
    """
    Copula-Base Outlier Detection Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics

    Returns:
        tuple: roc_auc score, f1 score and runtime of COPOD algorithm.
  """

    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = COPOD()
    model.fit(X)

    # Get the prediction labels and scores
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliners)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_copod = round(roc_auc_score(y, X_labels), 3)
    f1_score_copod = round(f1_score(y, X_labels, average='weighted'), 3)
    runtime_copod = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for COPOD model are: \n'
          f'ROC AUC: {roc_auc_copod}\n'
          f'F1 score: {f1_score_copod}\n'
          f'Time elapsed: {runtime_copod}')

    return roc_auc_copod, f1_score_copod, runtime_copod

# Define function for ECOD (Empirical Cumulative Outlier Detection) Algorithm
def model_ecod(X, y):
    """
    ECOD Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics

    Returns:
        tuple: roc_auc score, f1 score and runtime of ECOD algorithm.
  """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = ECOD()
    model.fit(X)

    # Get the prediction labels and scores
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliners)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_ecod = round(roc_auc_score(y, X_labels), 3)
    f1_score_ecod = round(f1_score(y, X_labels, average='weighted'), 3)
    runtime_ecod = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for ECOD model are: \n'
          f'ROC AUC: {roc_auc_ecod}\n'
          f'F1 score: {f1_score_ecod}\n'
          f'Time elapsed: {runtime_ecod}')

    return roc_auc_ecod, f1_score_ecod, runtime_ecod

def model_dbscan(X, y, eps, min_samples):
    """
    DBSCAN Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics

    Returns:
        tuple: roc_auc score, f1 score and runtime of DBSCAN algorithm.
    """
    # Record start time
    start_time = time.time()

    # Define model and its parameters
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    model.fit(X)

    # Get the prediction labels
    X_labels = model.labels_

    # Convert DBSCAN labels (-1, 1) to (1, 0)
    X_label = np.where(X_labels == -1, 1, 0)

    # Evaluation Metrics
    roc_auc_dbscan = round(roc_auc_score(y, X_label), 3)
    f1_score_dbscan = round(f1_score(y, X_label, average='weighted'), 3)
    runtime_dbscan = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for DBSCAN model are: \n'
          f'ROC AUC: {roc_auc_dbscan}\n'
          f'F1 score: {f1_score_dbscan}\n'
          f'Time elapsed: {runtime_dbscan}')

    return roc_auc_dbscan, f1_score_dbscan, runtime_dbscan


