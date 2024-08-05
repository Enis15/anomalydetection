from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

f1_scores = [] # To store the f1 scores for each fold
roc_auc_scores = [] # To store the roc auc scores for each fold

kf = KFold(n_splits=5, random_state=42, shuffle=True)

# Perform cross validation manually
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define the model
    clf = LocalOutlierFactor(n_neighbors=10)

    # Predict the labels
    y_pred = clf.fit_predict(X_test)

    # Convert LOF labels (-1, 1) to (1, 0)
    y_pred = (y_pred == -1).astype(int)

    # Calculate the performance scores and appended to the list
    roc_auc = roc_auc_score(y_test, y_pred)
    roc_auc_scores.append(roc_auc)

    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

# Calculate the mean metrics score for all folds.
mean_roc_auc = np.mean(roc_auc_scores)
mean_f1 = np.mean(f1_scores)

print(f'ROC AUC scores for each fold: {roc_auc_scores} \n '
      f'F1 scores for each fold: {f1_scores}')

print(f'Mean ROC AUC: {mean_roc_auc} & Mean F1 Score: {mean_f1}')