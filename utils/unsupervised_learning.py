import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold

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
        tuple: roc_auc score, f1 score and runtime of LOF algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the lists to store the metrics for each fold
    f1_scores = []
    roc_auc_scores = []

    # Fold details
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the model and the parameters
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', n_jobs=-1)
        # Fit the model
        clf.fit(X_train)
        # Predict the labels
        y_pred = clf.fit_predict(X_test)  # Outlier labels (1 = outliers & -1 = inliners)
        y_pred = (y_pred == -1).astype(int) # Convert LOF labels (-1, 1) to (1, 0)
        # Get the decision function scores
        y_score = -clf.negative_outlier_factor_
        #Calculate the performance scores and append to the list
        roc_auc_scores.append(roc_auc_score(y_test, y_score, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate the mean metrics for all folds
    roc_auc_lof = round(np.mean(roc_auc_scores), 3)
    f1_score_lof = round(np.mean(f1_scores), 3)
    runtime_lof = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for LOF model, with n_neighbors = {n_neighbors}, are: \n"
            f"ROC AUC: {roc_auc_scores} & Average ROC AUC Score {roc_auc_lof}\n"
            f"F1 score: {f1_scores} & Average F1 Score {f1_score_lof}\n" 
            f"Time elapsed: {runtime_lof} (s)")
    return roc_auc_lof, f1_score_lof, runtime_lof

'''
=======================================================
Define function for Isolation  Forest Algorithm
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
        tuple: roc_auc score, f1 score and runtime of Isolation Forest algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the lists to store the metrics for each fold
    f1_scores = []
    roc_auc_scores = []

    # Fold details
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the model and the parameters
        clf = IsolationForest(n_estimators=n_estimators, random_state=42)
        # Fit the model
        clf.fit(X_train)
        # Predict the labels
        y_pred = clf.predict(X_test) # Outlier labels (1 = outliers & -1 = inliners)
        y_pred = (y_pred == -1).astype(int) # Convert the labels (-1, 1) to (0, 1)
        # Calculate the decision scores
        y_score = clf.decision_function(X_test) # Raw label scores
        # Calculate the performance score and append them to the lists
        roc_auc_scores.append(roc_auc_score(y_test, y_score, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate the mean metrics for all folds
    roc_auc_iforest = round(np.mean(roc_auc_scores), 3)
    f1_score_iforest = round(np.mean(f1_scores), 3)
    runtime_iforest = round(time.time() - start_time, 3)


    print(f"Evaluation metrics for Isolation Forest model, with n_estimators = {n_estimators}, are: \n"
            f"ROC AUC: {roc_auc_scores} & Average ROC AUC Score {roc_auc_iforest}\n"
            f"F1 score: {f1_scores} & Average F1 Score {f1_score_iforest}\n" 
            f"Time elapsed: {runtime_iforest} (s)")
    return roc_auc_iforest, f1_score_iforest, runtime_iforest

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
        tuple: roc_auc score, f1 score and runtime of PCA algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the lists to store the metrics for each fold
    f1_scores = []
    roc_auc_scores = []

    # Fold details
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the model and the parameters
        clf = PCA()
        # Fit the model
        clf.fit(X_train)
        # Predict the labels
        y_pred = clf.predict(X_test)  # Labels (0, 1)
        # Calculate the decision scores
        y_score = clf.decision_function(X_test)  # Raw label scores
        # Calculate the performance score and append them to the lists
        roc_auc_scores.append(roc_auc_score(y_test, y_score, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate the mean metrics for all folds
    roc_auc_pca = round(np.mean(roc_auc_scores), 3)
    f1_score_pca = round(np.mean(f1_scores), 3)
    runtime_pca = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for PCA model are: \n"
            f"ROC AUC: {roc_auc_scores} & Average ROC AUC Score {roc_auc_pca}\n"
            f"F1 score: {f1_scores} & Average F1 Score {f1_score_pca}\n"
            f"Time elapsed: {runtime_pca} (s)")
    return roc_auc_pca, f1_score_pca, runtime_pca

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
        tuple: roc_auc score, f1 score and runtime of COPOD algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the lists to store the metrics for each fold
    f1_scores = []
    roc_auc_scores = []

    # Fold details
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the model and the parameters
        clf = COPOD()
        # Fit the model
        clf.fit(X_train)
        # Predict the labels
        y_pred = clf.predict(X_test)  # Labels (0, 1)
        # Calculate the decision function scores
        y_score = clf.decision_function(X_test)  # Raw label scores
        # Calculate the performance score and append them to the lists
        roc_auc_scores.append(roc_auc_score(y_test, y_score, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate the mean metrics for all folds
    roc_auc_copod = round(np.mean(roc_auc_scores), 3)
    f1_score_copod = round(np.mean(f1_scores), 3)
    runtime_copod = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for COPOD model are: \n"
            f"ROC AUC: {roc_auc_scores} & Average ROC AUC Score {roc_auc_copod}\n"
            f"F1 score: {f1_scores} & Average F1 Score {f1_score_copod}\n"
            f"Time elapsed: {runtime_copod} (s)")
    return roc_auc_copod, f1_score_copod, runtime_copod

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
        tuple: roc_auc score, f1 score and runtime of ECOD algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the lists to store the metrics for each fold
    f1_scores = []
    roc_auc_scores = []

    # Fold details
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define the model and the parameters
        clf = ECOD()
        # Fit the model
        clf.fit(X)
        # Predict the labels
        y_pred = clf.predict(X_test)  # Labels (0, 1)
        # Calculate the decision function scores
        y_score = clf.decision_function(X_test)  # Raw label scores
        # Calculate the performance score and append them to the lists
        roc_auc_scores.append(roc_auc_score(y_test, y_score, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate the mean metrics for all folds
    roc_auc_ecod = round(np.mean(roc_auc_scores), 3)
    f1_score_ecod= round(np.mean(f1_scores), 3)
    runtime_ecod = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for ECOD model are: \n"
            f"ROC AUC: {roc_auc_scores} & Average ROC AUC Score {roc_auc_ecod}\n"
            f"F1 score: {f1_scores} & Average F1 Score {f1_score_ecod}\n"
            f"Time elapsed: {runtime_ecod} (s)")
    return roc_auc_ecod, f1_score_ecod, runtime_ecod

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
        tuple: roc_auc score, f1 score and runtime of DBSCAN algorithm.
    """
    # Record start time
    start_time = time.time()

    # Define the lists to store the metrics for each fold
    f1_scores = []
    roc_auc_scores = []

    # Fold details
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Define model and its parameters
        clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        # Fit the model
        clf.fit(X_train)
        # Get the prediction labels
        y_pred = clf.fit_predict(X_test) # Outlier labels (1 = outliers & -1 = inliners)
        y_pred = (y_pred == -1).astype(int) # Convert labels (-1, 1) to (1, 0)
        # Calculate the performance scores and append to the list
        roc_auc_scores.append(roc_auc_score(y_test, y_pred, average='weighted')) # using y_pred since DBSCAN doesn't have 'decision_function'

        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

    # Calculate the mean metrics for all folds
    roc_auc_dbscan = round(np.mean(roc_auc_scores), 3)
    f1_score_dbscan = round(np.mean(f1_scores), 3)
    runtime_dbscan = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for DBSCAN model are: \n"
            f"ROC AUC: {roc_auc_scores} & Average ROC AUC Score{roc_auc_dbscan}\n"
            f"F1 score: {f1_scores} & Average F1 Score {f1_score_dbscan}\n"
            f"Time elapsed: {runtime_dbscan} (s)")
    return roc_auc_dbscan, f1_score_dbscan, runtime_dbscan


