import os
import time
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from multiprocessing import freeze_support
from adjustText import adjust_text
import matplotlib.pyplot as plt

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.cluster import DBSCAN  # Use Scikit-learn's DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# Import the logger function
from utils.logger import logger


# Initialize the logger
_logger = logger(__name__)

def model_dbscan(X, y, eps, min_samples):
    """
    DBSCAN Algorithm for anomaly detection.
    Parameters:
        X: Input Features.
        y: True Labels.
        eps: Distance threshold.
        min_samples: Minimum number of samples required to form a cluster.
    Returns:
        tuple: roc_auc score, f1 score, precision score, recall score, accuracy score, and runtime for DBSCAN algorithm.
    """
    start_time = time.time()

    pca = PCA(n_components=0.7)
    X_pca = pca.fit_transform(X)

    # Use Scikit-learn's DBSCAN
    clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)

    # Fit the model
    y_pred = clf.fit_predict(X)
    y_pred = (y_pred == -1).astype(int)

    # Calculate the performance scores
    roc_auc_dbscan = round(roc_auc_score(y, y_pred, average='weighted'), 3)
    f1_score_dbscan = round(f1_score(y, y_pred, average='weighted'), 3)
    precision_dbscan = round(precision_score(y, y_pred, average='weighted'), 3)
    recall_dbscan = round(recall_score(y, y_pred, average='weighted'), 3)
    accuracy_dbscan = round(balanced_accuracy_score(y, y_pred), 3)
    runtime_dbscan = round((time.time() - start_time), 3)

    print(f"Evaluation metrics for DBSCAN model: \n"
          f"ROC AUC: {roc_auc_dbscan}\nF1 Score: {f1_score_dbscan}\n"
          f"Precision: {precision_dbscan}\nRecall: {recall_dbscan}\n"
          f"Accuracy: {accuracy_dbscan}\nTime elapsed: {runtime_dbscan} (s)")

    return roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan



# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/Fraud.csv')
print(df.shape)
df = df.sample(frac=0.5, random_state=42)
print(df.shape)
# Feature engineering: Dropping the columns 'nameOrig' & 'nameDest'; Encoding values to the column 'CASH_OUT'
df = df.drop(['nameOrig', 'nameDest'], axis=1)
df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

# Determining the X and y values
X = df.drop('isFraud', axis=1)
y = df['isFraud'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardize the data


if __name__ == '__main__':
    freeze_support()

    # DataFrame to store the evaluation metrics
    metrics = []
    metrics_unsupervised = []


    def append_metrics(modelname, estimator, roc_auc, f1_score, runtime):
        metrics.append({
            'Model': modelname,
            'Estimator': estimator,
            'ROC_AUC_Score': roc_auc,
            'F1_Score': f1_score,
            'Runtime': runtime
        })


    def unsupervised_metrics(modelname, estimator, roc_auc, f1_score, precision, recall, accuracy, runtime):
        metrics_unsupervised.append({
            'Model': modelname,
            'Estimator': estimator,
            'ROC_AUC_Score': roc_auc,
            'F1_Score': f1_score,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'Runtime': runtime
        })


    try:
        _logger.info('Starting DBSCAN Evaluation')
        best_eps = 0.7021425484240491
        best_min_samples = 31
        roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan = model_dbscan(X_scaled, y, eps=best_eps, min_samples=best_min_samples)
        append_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, runtime_dbscan)
        unsupervised_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan)
        _logger.info(f'DBSCAN Evaluation: ROC AUC={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, Precision={precision_dbscan}, Recall={recall_dbscan}, Accuracy={accuracy_dbscan}, Runtime={runtime_dbscan}')
    except Exception as e:
        _logger.error(f'Error evaluating DBSCAN model: {e}')
