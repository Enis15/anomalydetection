import time
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate, KFold

# Import the models
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Determine the metrics for performance evaluation
scorer = {'f1_score': make_scorer(f1_score), 'roc_auc': make_scorer(roc_auc_score)}

# Determine the folds for the cross validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

'''
=======================================================
Define function for KNN Algorithm
=======================================================
'''
def model_knn(X, y, k, scorer, kf):
    """
    KNN Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        k: number of neighbors.
        scorer: dict containing performance metrics.
        kf: KFold object.
    Returns:
        tuple: roc_auc score, f1 score and runtime of KNN algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = KNeighborsClassifier(n_neighbors=k, metric='minkowski', n_jobs=-1)

    # Evaluation metrics for each fold
    results = cross_validate(estimator=clf, X=X, y=y, cv=kf, scoring=scorer)

    #Calculate the mean metrics
    roc_auc_knn = round(results['test_roc_auc'].mean(), 3)
    f1_knn = round(results['test_f1_score'].mean(), 3)
    runtime_knn = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for KNN model, with k = {k}, are: \n"
            f"ROC AUC: {results['test_roc_auc']} & Average ROC AUC {roc_auc_knn}\n"
            f"F1 score: {results['test_f1_score']} & Average ROC AUC {f1_knn}\n" 
            f"Time elapsed: {runtime_knn} (s)")
    return roc_auc_knn, f1_knn, runtime_knn

'''
=======================================================
Define function for XGBoost Algorithm
=======================================================
'''
def model_xgboost(X, y, n_estimators, max_depth, learning_rate, scorer, kf):
    """
    XGBoost Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        n_estimators: number of trees.
        max_depth: maximum depth of tree.
        learning_rate: learning rate.
        scorer: dict containing performance metrics.
        kf: KFold object.
    Returns:
        tuple: roc_auc score, f1 score and runtime of XGBooost algorithm.
    """
    # Record the start times
    start_time = time.time()

    # Create a dictionary with the parameters needed to initiate the classifier
    params = {
        'objective': 'binary:logistic',
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
    }
    # Define the model and the parameters
    clf = XGBClassifier(**params)

    # Evaluation metrics for each fold
    results = cross_validate(estimator=clf, X=X, y=y, cv=kf, scoring=scorer)

    # Calculate the mean metrics
    roc_auc_xgboost = round(results['test_roc_auc'].mean(), 3)
    f1_score_xgboost = round(results['test_f1_score'].mean(), 3)
    runtime_xgboost = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for XGBoost model, are: \n"
            f"ROC AUC: {results['test_roc_auc']} & Average ROC AUC {roc_auc_xgboost}\n"
            f"ROC AUC: {results['test_f1_score']} & Average ROC AUC {f1_score_xgboost}\n"
            f"Time elapsed: {runtime_xgboost} (s)")
    return roc_auc_xgboost, f1_score_xgboost, runtime_xgboost

'''
=========================================================
Define function for SVM (Support Vector Machine) Algorithm
=========================================================
'''
def model_svm(X, y, scorer, kf):
    """
    SVM Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: Label class.
        scorer: dict containing performance metrics.
        kf: KFold object.
    Returns:
        tuple: roc_auc score, f1 score and runtime of SVM algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf  = svm.SVC()

    # Evaluation metrics for each fold
    results = cross_validate(estimator=clf, X=X, y=y, cv=kf, scoring=scorer)

    # Calculate the mean metrics
    roc_auc_svm = round(results['test_roc_auc'].mean(), 3)
    f1_score_svm = round(results['test_f1_score'].mean(), 3)
    runtime_svm = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for SVM model are: \n"
            f"ROC AUC: {results['test_roc_auc']} & Average ROC AUC {roc_auc_svm}\n"
            f"F1 score: {results['test_f1_score']} & Average ROC AUC {f1_score_svm}\n" 
            f"Time elapsed: {runtime_svm} (s)")
    return roc_auc_svm, f1_score_svm, runtime_svm

'''
=======================================================
Define function for Naive Bayes Algorithm
=======================================================
'''

def model_nb(X, y, scorer, kf):
    """
    Naive Bayes Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: Label class.
        scorer: dict containing performance metrics.
        kf: KFold object.
    Returns:
        tuple: roc_auc score, f1 score and runtime of Naive Bayes algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = GaussianNB()

    # Evaluation metrics for each fold
    results = cross_validate(estimator=clf, X=X, y=y, cv=kf, scoring=scorer)
    # Calculate the mean metrics
    roc_auc_nb = round(results['test_roc_auc'].mean(), 3)
    f1_score_nb = round(results['test_f1_score'].mean(), 3)
    runtime_nb = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for Naive Bayes model are: \n"
            f"ROC AUC: {results['test_roc_auc']} & Average ROC AUC {roc_auc_nb}\n"
            f"F1 score: {results['test_f1_score']} & Average ROC AUC {f1_score_nb}\n" 
            f"Time elapsed: {runtime_nb} (s)")
    return roc_auc_nb, f1_score_nb, runtime_nb

'''
=======================================================
Define function for Random Forest Classifier Algorithm
=======================================================
'''
def model_rf(X, y, n_estimators, max_depth, scorer, kf):
    """
    Random Forest Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        n_estimators: number of trees.
        max_depth: maximum depth of tree.
        scorer: dict containing performance metrics.
        kf: KFold object.
    Returns:
        tuple: roc_auc score, f1 score and runtime of Naive Bayes algorithm.
    """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)

    # Evaluation metrics for each fold
    results = cross_validate(estimator=clf, X=X, y=y, cv=kf, scoring=scorer)

    # Calculate the mean metrics
    roc_auc_rf = round(results['test_roc_auc'].mean(), 3)
    f1_score_rf = round(results['test_f1_score'].mean(), 3)
    runtime_rf = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for Random Forest Classifier are: \n"
            f"ROC AUC: {results['test_roc_auc']} & Average ROC AUC {roc_auc_rf}\n"
            f"F1 score: {results['test_f1_score']} & Average ROC AUC {f1_score_rf}\n" 
            f"Time elapsed: {runtime_rf} (s)")
    return roc_auc_rf, f1_score_rf, runtime_rf

'''
=======================================================
Define function for CatBoost Algorithm
=======================================================
'''
def model_cb(X, y, iterations, learning_rate, depth, scorer, kf):
    """
        CatBoost Algorithm for anomaly detection.

        Parameters:
            X: Input Features.
            y: True Labels.
            iterations: The number of iterations.
            learning_rate: The learning rate.
            depth: The depth of the tree.
            scorer: dict containing performance metrics.
            kf: KFold object.
        Returns:
            tuple: roc_auc score, f1 score and runtime of CatBoost Algorithm.
        """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    clf = CatBoostClassifier(iterations=iterations,
                                learning_rate=learning_rate,
                                depth=depth,
                                verbose=False)

    # Evaluation metrics for each fold
    results = cross_validate(estimator=clf, X=X, y=y, cv=kf, scoring=scorer)

    # Calculate the mean metrics
    roc_auc_cb = round(results['test_roc_auc'].mean(), 3)
    f1_score_cb = round(results['test_f1_score'].mean(), 3)
    runtime_cb = round(time.time() - start_time, 3)

    print(f"Evaluation metrics for CatBoost are: \n"
            f"ROC AUC: {results['test_roc_auc']} & Average ROC AUC {roc_auc_cb}\n"
            f"F1 score: {results['test_f1_score']} & Average ROC AUC {f1_score_cb}\n" 
            f"Time elapsed: {runtime_cb} (s)")
    return roc_auc_cb, f1_score_cb, runtime_cb