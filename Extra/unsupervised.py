import time
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from utils.paramet_tune import LOF_tuner

def model_lof(X, y, n_neighbors):
    """
    LOF Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        n_neighbors: number of neighbors.
    Returns:
        tuple: roc_auc score, f1 score, NMI index, adjusted rand score and runtime of LOF algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', n_jobs=-1)
    # Fit the model
    clf.fit(X)
    # Predict the labels
    y_pred = clf.fit_predict(X)  # Outlier labels (1 = outliers & -1 = inliners)
    y_pred = (y_pred == -1).astype(int) # Convert LOF labels (-1, 1) to (1, 0)
    # Get the decision function scores
    y_score = -clf.negative_outlier_factor_

    #Calculate the performance scores
    roc_auc_lof = round(roc_auc_score(y, y_score, average='weighted'), 3)
    f1_score_lof = round(f1_score(y, y_pred, average='weighted'), 3)
    nmi_score_lof = round(normalized_mutual_info_score(y, y_pred), 3)
    rand_score_lof = round(adjusted_rand_score(y, y_pred), 3)
    runtime_lof =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for LOF model, with n_neighbors = {n_neighbors}, are: \n"
            f"ROC AUC: {roc_auc_lof}\n"
            f"F1 score: {f1_score_lof}\n"
            f"NMI score: {nmi_score_lof}\n"
            f"Adjusted Rand Score: {rand_score_lof}\n"
            f"Time elapsed: {runtime_lof} (s)")
    return roc_auc_lof, f1_score_lof, nmi_score_lof, rand_score_lof, runtime_lof

def model_dbscan(X, y, eps, min_samples):
    """
    DBSCAN Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        eps: Distance threshold.
        min_samples: Minimum number of samples required to form a cluster.
    Returns:
        tuple: roc_auc score, f1 score and runtime of DBSCAN algorithm.
    """
    # Record start time
    start_time = time.time()

    # Define model and its parameters
    clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    # Fit the model
    clf.fit(X)
    # Get the prediction labels
    y_pred = clf.fit_predict(X) # Outlier labels (1 = outliers & -1 = inliners)
    y_pred = (y_pred == -1).astype(int) # Convert labels (-1, 1) to (1, 0)

    #Calculate the performance scores
    roc_auc_dbscan = round(roc_auc_score(y, y_pred, average='weighted'), 3)
    f1_score_dbscan = round(f1_score(y, y_pred, average='weighted'), 3)
    nmi_score_dbscan = round(normalized_mutual_info_score(y, y_pred), 3)
    rand_score_dbscan = round(adjusted_rand_score(y, y_pred), 3)
    runtime_dbscan =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for DBSCAN model are: \n"
            f"ROC AUC: {roc_auc_dbscan}\n"
            f"F1 score: {f1_score_dbscan}\n"
            f"NMI score: {nmi_score_dbscan}\n"
            f"Adjusted Rand Score: {rand_score_dbscan}\n"
            f"Time elapsed: {runtime_dbscan} (s)")
    return roc_auc_dbscan, f1_score_dbscan, nmi_score_dbscan, rand_score_dbscan, runtime_dbscan

df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

# Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

smote = SMOTE(random_state=42)
X_re, y_re = smote.fit_resample(X, y)

scaler = StandardScaler()
X = scaler.fit_transform(X) # Standardize the data

lof_tune = LOF_tuner(X, y)
k_lof = lof_tune.tune_model()
roc_auc, f1_score, nmi_score, rand_score, runtime = model_lof(X, y, k_lof)

# rand_score ==> ari_score