import os
import time
import pandas as pd
from dask.graph_manipulation import chunks
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from multiprocessing import freeze_support
from adjustText import adjust_text
import matplotlib.pyplot as plt
import dask.array as da
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.cluster import DBSCAN  # Use Scikit-learn's DBSCAN
from sklearn.decomposition import PCA

# Import the logger function
from utils.logger import logger

# Hyperparameter tuning functions
from utils.paramet_tune import DBSCAN_tuner

# Initialize the logger
_logger = logger(__name__)


class DBSCAN_tuner:
    """
    Hyperparameter tuning for DBSCAN classifier.
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
    Methods:
        - objective(params): Defines the optimization objective for hyperparameter tuning.
        - tune_model(): Performs hyperparameter tuning using Bayesian optimization.
    Example usage:
    dbscan_tuner = DBSCAN_tuner(X, y)
    best_params = dbscan_tuner.tune_model()  --> Run the final DBSCAN model with optimal parameters.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(self, params):
        eps = params['eps']
        min_samples = int(params['min_samples'])

        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(self.X)

        # Define DBSCAN model and its parameters
        clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        # Fit and predict
        y_pred = clf.fit_predict(X_pca)
        y_pred = (y_pred == -1).astype(int)  # Convert labels (-1, 1) to (1, 0)

        # Calculate the performance scores
        roc_auc = round(roc_auc_score(self.y, y_pred, average='weighted'), 3)

        return {'loss': -roc_auc, 'status': STATUS_OK, 'roc_auc_score': roc_auc}

    def tune_model(self):
        space = {
            'eps': hp.uniform('eps', 0.2, 0.9),
            'min_samples': hp.quniform('min_samples', 10, 100, 1)
        }
        trials = Trials()
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)
        return best


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

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X)

    # Use Scikit-learn's DBSCAN
    clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)

    # Fit the model
    y_pred = clf.fit_predict(X_pca)
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
df = pd.read_csv('../data/datasets/Labeled_DS/fin_paysys/finpays.csv')

# Encoding categorical values with numerical variables
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

# Determine the X and y values
X = df.drop('fraud', axis=1)
y = df['fraud'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_dask = da.from_array(X_scaled, chunks=(100000, X.shape[1]))

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

        dbscan_tuner = DBSCAN_tuner(X_dask, y)
        dbscan_cluster = dbscan_tuner.tune_model()
        _logger.info(f'Best DBSCAN Model: {dbscan_cluster}')

        best_eps = dbscan_cluster['eps']
        best_min_samples = int(dbscan_cluster['min_samples'])

        roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan = model_dbscan(
            X_dask, y, eps=best_eps, min_samples=best_min_samples)

        append_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, runtime_dbscan)
        unsupervised_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan,
                             accuracy_dbscan, runtime_dbscan)

        _logger.info(f'DBSCAN Evaluation: ROC AUC={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, '
                     f'Precision={precision_dbscan}, Recall={recall_dbscan}, Accuracy={accuracy_dbscan}, Runtime={runtime_dbscan}')

    except Exception as e:
        _logger.error(f'Error evaluating DBSCAN model: {e}')
