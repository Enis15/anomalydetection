from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Define the lists to store the metrics for each fold
f1_scores = []
roc_auc_scores = []

# Fold details
kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define the model
    clf = LocalOutlierFactor(n_neighbors=10)

    # Fit the model
    clf.fit(X_train)

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

# Define the lists to store the metrics for each fold
f1_scores = []
roc_auc_scores = []

# Fold details
kf = KFold(n_splits=5, random_state=42, shuffle=True)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define the model and the parameters
    clf = IForest(n_estimators=n_estimators, random_state=42)

    # Fit the model
    clf.fit(X_train)

    # Predict the labels
    y_pred = clf.predict(X_test)  # Labels (0, 1)
    y_score = clf.decision_function(X_test)  # Raw label scores

    # Calculate the performance score and append them to the lists
    roc_auc_scores.append(roc_auc_score(y_pred, y_test))
    f1_scores.append(f1_score(y_pred, y_test))

# Calculate the mean metrics for all folds
roc_auc_iforest = round(np.mean(roc_auc_scores), 3)
f1_score_iforest = round(np.mean(f1_scores), 3)
runtime_iforest = round(time.time() - start_time, 3)

print(f"Evaluation metrics for LOF model, with n_estimators = {n_estimators}, are: \n"
      f"ROC AUC: {roc_auc_scores} & Average ROC AUC {roc_auc_iforest}\n"
      f"F1 score: {f1_scores} & Average ROC AUC {f1_score_iforest}\n"
      f"Time elapsed: {runtime_iforest} (s)")
return roc_auc_iforest, f1_score_iforest, runtime_iforest

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

    # Convert LOF labels (-1, 1) to (1, 0)
    y_pred = (y_pred == -1).astype(int)

    # Calculate the performance scores and append to the list
    roc_auc_scores.append(roc_auc_score(y_pred, y_test))
    f1_scores.append(f1_score(y_pred, y_test))

# Calculate the mean metrics for all folds
roc_auc_lof = round(np.mean(roc_auc_scores), 3)
f1_score_lof = round(np.mean(f1_scores), 3)
runtime_lof = round(time.time() - start_time, 3)

print(f"Evaluation metrics for LOF model, with n_neighbors = {n_neighbors}, are: \n"
      f"ROC AUC: {roc_auc_scores} & Average ROC AUC {roc_auc_lof}\n"
      f"F1 score: {f1_scores} & Average ROC AUC {f1_score_lof}\n"
      f"Time elapsed: {runtime_lof} (s)")
return roc_auc_lof, f1_score_lof, runtime_lof