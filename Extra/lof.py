import os
import time
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import gen_batches

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.cluster import DBSCAN  # Use Scikit-learn's DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# Import the logger function
from utils.logger import logger

# Initialize the logger
_logger = logger(__name__)

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/Fraud.csv')
print(df.shape)

# Feature engineering: Dropping the columns 'nameOrig' & 'nameDest'; Encoding values to the column 'CASH_OUT'
df = df.drop(['nameOrig', 'nameDest'], axis=1)
df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

# Determining the X and y values
X = df.drop('isFraud', axis=1)
y = df['isFraud'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardize the data

best_eps = 0.70215
best_min_samples = 31

_logger.info('Starting DBSCAN Evaluation')

start_time = time.time()

batches = list(gen_batches(X_scaled.shape[0], 300000))

# Initialize scores
roc_auc_scores = []
f1_scores = []
precision_scores = []
recall_scores = []
accuracy_scores = []
total_runtime = 0

# Iterate over batches
for i in batches:

      X_batch = X_scaled[i]
      y_batch = y[i]

      # Fit DBSCAN on each batch
      clf = DBSCAN(eps=best_eps, min_samples=best_min_samples, metric='euclidean', n_jobs=-1)
      y_pred_batch = clf.fit_predict(X_batch)

      # Convert labels
      y_pred_batch = (y_pred_batch == -1).astype(int)

      # Calculate metrics for current batch
      roc_auc_batch = round(roc_auc_score(y_batch, y_pred_batch, average='weighted'), 3)
      f1_score_batch = round(f1_score(y_batch, y_pred_batch, average='weighted'), 3)
      precision_batch = round(precision_score(y_batch, y_pred_batch, average='weighted'), 3)
      recall_batch = round(recall_score(y_batch, y_pred_batch, average='weighted'), 3)
      accuracy_batch = round(balanced_accuracy_score(y_batch, y_pred_batch), 3)
      runtime_batch = round((time.time() - start_time), 3)

      # Store the results
      roc_auc_scores.append(roc_auc_batch)
      f1_scores.append(f1_score_batch)
      precision_scores.append(precision_batch)
      recall_scores.append(recall_batch)
      accuracy_scores.append(accuracy_batch)
      total_runtime += runtime_batch

      # Print the results for the batch
      print(f'Batch Results: \n'
            f'ROC AUC: {roc_auc_batch}\n'
            f'F1 Score: {f1_score_batch}\n'
            f'Precision: {precision_batch}\n'
            f'Recall: {recall_batch}\n'
            f'Accuracy: {accuracy_batch}\n'
            f'Runtime: {runtime_batch}')

# Average results across all batches
roc_auc_dbscan = round(sum(roc_auc_batch) / len(roc_auc_batch), 3)
f1_score_dbscan = round(sum(f1_score_batch) / len(f1_score_batch), 3)
precision_dbscan = round(sum(precision_batch) / len(precision_batch), 3)
recall_dbscan = round(sum(recall_batch) / len(recall_batch), 3)
accuracy_dbscan = round(balanced_accuracy_score(y_batch, y_pred_batch), 3)
runtime_dbscan = round(total_runtime / len(batches), 3)

print(f"Evaluation metrics for DBSCAN model: \n"
      f"ROC AUC: {roc_auc_dbscan}\n"
      f"F1 Score: {f1_score_dbscan}\n"
      f"Precision: {precision_dbscan}\n"
      f"Recall: {recall_dbscan}\n"
      f"Accuracy: {accuracy_dbscan}\n"
      f"Time elapsed: {runtime_dbscan} (s)")

_logger.info(
      f'DBSCAN Evaluation: ROC AUC={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, Precision={precision_dbscan}, Recall={recall_dbscan}, Accuracy={accuracy_dbscan}, Runtime={runtime_dbscan}')
