import pandas as pd
from pyod.models.lof import LOF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
import time

df = pd.read_csv('../data/datasets/Unlabeled_DS/CreditCardTransaction.csv')

X = df[['TrnxAmount']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LOF(n_neighbors=7, contamination=0.1)
model.fit(X_scaled)

threshold = 1
#y_pred = model.lables_
y_scores = model.decision_scores_
y_pred = [1 if score > threshold else 0 for score in y_scores]

print(f'Predicted labels (1: Anomaly, 0: Inlier):')
print(y_pred)