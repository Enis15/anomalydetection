import time
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score

# Import the models
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from sklearn.cluster import DBSCAN

'''
=======================================================
Define function for LOF (Local Outlier Factor) Algorithm
=======================================================
'''
def model_lof(X, y, n_neighbors):
    """
    LOF Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        n_neighbors: number of neighbors.
    Returns:
        tuple: roc_auc score, f1 score, precision score, recall score, accuracy score and runtime for LOF algorithm.
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
    precision_lof = round(precision_score(y, y_pred, average='weighted'), 3)
    recall_lof = round(recall_score(y, y_pred, average='weighted'), 3)
    accuracy_lof = round(balanced_accuracy_score(y, y_pred), 3)
    runtime_lof =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for LOF model, with n_neighbors = {n_neighbors}, are: \n"
            f"ROC AUC: {roc_auc_lof}\n"
            f"F1 score: {f1_score_lof}\n"
            f"Precision score: {precision_lof}\n"
            f"Recall score: {recall_lof}\n"
            f"Accuracy score: {accuracy_lof}\n"
            f"Time elapsed: {runtime_lof} (s)")
    return roc_auc_lof, f1_score_lof, precision_lof, recall_lof, accuracy_lof, runtime_lof

'''
=======================================================
Define function for Isolation Forest Algorithm
=======================================================
'''
# Define function for IForest (Isolation Forest) Algorithm
def model_iforest(X, y, n_estimators):
    """
    Isolation Forest Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        n_estimators: Number of trees in the forest.
    Returns:
        tuple: roc_auc score, f1 score, precision score, recall score, accuracy score and runtime for Isolation Forest algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = IsolationForest(n_estimators=n_estimators, random_state=42)
    # Fit the model
    clf.fit(X)
    # Predict the labels
    y_pred = clf.predict(X) # Outlier labels (1 = outliers & -1 = inliners)
    y_pred = (y_pred == -1).astype(int) # Convert the labels (-1, 1) to (0, 1)
    # Calculate the decision scores
    y_score = clf.decision_function(X) # Raw label scores

    #Calculate the performance scores
    roc_auc_if = round(roc_auc_score(y, y_score, average='weighted'), 3)
    f1_score_if = round(f1_score(y, y_pred, average='weighted'), 3)
    precision_if = round(precision_score(y, y_pred, average='weighted'), 3)
    recall_if = round(recall_score(y, y_pred, average='weighted'), 3)
    accuracy_if = round(balanced_accuracy_score(y, y_pred), 3)
    runtime_if =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for Isolation Forest model, with n_estimators = {n_estimators}, are: \n"
            f"ROC AUC: {roc_auc_if}\n"
            f"F1 score: {f1_score_if}\n"
            f"Precision score: {precision_if}\n"
            f"Recall score: {recall_if}\n"
            f"Accuracy score: {accuracy_if}\n"
            f"Time elapsed: {runtime_if} (s)")
    return roc_auc_if, f1_score_if, precision_if, recall_if, accuracy_if, runtime_if

'''
===============================================================
Define function for PCA (Principal Component Analysis) Algorithm
===============================================================
'''
def model_pca(X, y):
    """
    PCA Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
    Returns:
        tuple: roc_auc score, f1 score, precision score, recall score, accuracy score and runtime for PCA algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = PCA()
    # Fit the model
    clf.fit(X)
    # Predict the labels
    y_pred = clf.predict(X)  # Labels (0, 1)
    # Calculate the decision scores
    y_score = clf.decision_function(X)  # Raw label scores

    #Calculate the performance scores
    roc_auc_pca = round(roc_auc_score(y, y_score, average='weighted'), 3)
    f1_score_pca = round(f1_score(y, y_pred, average='weighted'), 3)
    precision_pca = round(precision_score(y, y_pred, average='weighted'), 3)
    recall_pca = round(recall_score(y, y_pred, average='weighted'), 3)
    accuracy_pca = round(balanced_accuracy_score(y, y_pred), 3)
    runtime_pca =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for Principal Component model are: \n"
            f"ROC AUC: {roc_auc_pca}\n"
            f"F1 score: {f1_score_pca}\n"
            f"Precision score: {precision_pca}\n"
            f"Recall score: {recall_pca}\n"
            f"Accuracy score: {accuracy_pca}\n"
            f"Time elapsed: {runtime_pca} (s)")
    return roc_auc_pca, f1_score_pca, precision_pca, recall_pca, accuracy_pca,runtime_pca

'''
===================================================================
Define function for COPOD (Copula Based Outlier Detection) Algorithm
===================================================================
'''
def model_copod(X, y):
    """
    Copula-Base Outlier Detection Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
    Returns:
        tuple:roc_auc score, f1 score, precision score, recall score, accuracy score and runtime for COPOD algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = COPOD()
    # Fit the model
    clf.fit(X)
    # Predict the labels
    y_pred = clf.predict(X)  # Labels (0, 1)
    # Calculate the decision function scores
    y_score = clf.decision_function(X)  # Raw label scores

    #Calculate the performance scores
    roc_auc_copod = round(roc_auc_score(y, y_score, average='weighted'), 3)
    f1_score_copod = round(f1_score(y, y_pred, average='weighted'), 3)
    precision_copod = round(precision_score(y, y_pred, average='weighted'), 3)
    recall_copod = round(recall_score(y, y_pred, average='weighted'), 3)
    accuracy_copod = round(balanced_accuracy_score(y, y_pred), 3)
    runtime_copod =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for COPOD model are: \n"
            f"ROC AUC: {roc_auc_copod}\n"
            f"F1 score: {f1_score_copod}\n"
            f"Precision score: {precision_copod}\n"
            f"Recall score: {recall_copod}\n"
            f"Accuracy score: {accuracy_copod}\n"
            f"Time elapsed: {runtime_copod} (s)")
    return roc_auc_copod, f1_score_copod, precision_copod, recall_copod, accuracy_copod, runtime_copod


'''
===========================================================================
Define function for ECOD (Empirical Cumulative Outlier Detection) Algorithm
==========================================================================
'''
def model_ecod(X, y):
    """
    ECOD Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
    Returns:
        tuple: roc_auc score, f1 score, precision score, recall score, accuracy score and runtime for ECOD algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = ECOD()
    # Fit the model
    clf.fit(X)
    # Predict the labels
    y_pred = clf.predict(X)  # Labels (0, 1)
    # Calculate the decision function scores
    y_score = clf.decision_function(X)  # Raw label scores

    #Calculate the performance scores
    roc_auc_ecod = round(roc_auc_score(y, y_score, average='weighted'), 3)
    f1_score_ecod = round(f1_score(y, y_pred, average='weighted'), 3)
    precision_ecod = round(precision_score(y, y_pred, average='weighted'), 3)
    recall_ecod = round(recall_score(y, y_pred, average='weighted'), 3)
    accuracy_ecod = round(balanced_accuracy_score(y, y_pred), 3)
    runtime_ecod =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for ECOD model are: \n"
            f"ROC AUC: {roc_auc_ecod}\n"
            f"F1 score: {f1_score_ecod}\n"
            f"Precision score: {precision_ecod}\n"
            f"Recall score: {recall_ecod}\n"
            f"Accuracy score: {accuracy_ecod}\n"
            f"Time elapsed: {runtime_ecod} (s)")
    return roc_auc_ecod, f1_score_ecod, precision_ecod, recall_ecod, accuracy_ecod, runtime_ecod

'''
=======================================================
Define function for DBSCAN  Algorithm
=======================================================
'''
def model_dbscan(X, y, eps, min_samples):
    """
    DBSCAN Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        eps: Distance threshold.
        min_samples: Minimum number of samples required to form a cluster.
    Returns:
        tuple: roc_auc score, f1 score, precision score, recall score, accuracy score and runtime for DBSCAN algorithm.
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
    precision_dbscan = round(precision_score(y, y_pred, average='weighted'), 3)
    recall_dbscan = round(recall_score(y, y_pred, average='weighted'), 3)
    accuracy_dbscan = round(balanced_accuracy_score(y, y_pred), 3)
    runtime_dbscan =round((time.time() - start_time), 3)

    print(f"Evaluation metrics for DBSCAN model are: \n"
            f"ROC AUC: {roc_auc_dbscan}\n"
            f"F1 score: {f1_score_dbscan}\n"
            f"Precision score: {precision_dbscan}\n"
            f"Recall score: {recall_dbscan}\n"
            f"Accuracy score: {accuracy_dbscan}\n"
            f"Time elapsed: {runtime_dbscan} (s)")
    return roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan
