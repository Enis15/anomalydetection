import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import time

# datasets = [50000, 150000, 350000, 550000, 750000, 1000000]
#
#
# for dataset in datasets:
start = time.time()

X, y = make_classification(n_samples=50000, n_classes=2, n_features=25, random_state=42)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = normalize(X)

model = DBSCAN(eps=0.8, min_samples=150, metric='euclidean')
clf = model.fit(X)
labels = clf.labels_

labels = np.where(labels == -1, 1, 0)
print(labels)

roc_auc = roc_auc_score(y, labels)
print(roc_auc)
f1_score = f1_score(y, labels)
print(f1_score)
endtime = time.time() - start
print(endtime)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10,8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50)
plt.colorbar(scatter, label='Principal components')
plt.grid(True)
plt.show()

