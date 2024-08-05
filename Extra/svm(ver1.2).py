import os
import pandas as pd
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support
from adjustText import adjust_text
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, normalize

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from utils.paramet_tune import DBSCAN_tuner
"""
df = pd.read_csv('../data/datasets/Labeled_DS/Fraud.csv')

df = df.drop({'nameOrig', 'nameDest'}, axis=1)
df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

X = df.drop(['isFraud'], axis=1)
y = df['isFraud'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = normalize(X)
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DBSCAN(eps=0.3, min_samples=50)
X_labels = clf.fit(X)
lables = X_labels.labels_
"""

X, y = make_classification(n_samples=10000, n_features=5, n_classes=2, random_state=42)

dbscan_tuner = DBSCAN_tuner(X, y)
best_model = dbscan_tuner.tune_model()
b_eps = best_model['eps']
b_min_samples = int(best_model['min_samples'])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = normalize(X)
X = pd.DataFrame(X)

clf = DBSCAN(eps=b_eps, min_samples=b_min_samples).fit(X)
X_labels = clf.labels_

print(X_labels)

roc_auc = roc_auc_score(y, X_labels, average='weighted')
f1 = f1_score(y, X_labels, average='weighted')

print(roc_auc)
print(f1)
