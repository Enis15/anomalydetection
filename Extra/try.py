import pandas as pd
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
import time
import numpy as np

df = pd.read_csv('../data/datasets/Unlabeled_DS/CreditCardTransaction.csv')
X = df[['TrnxAmount']].values
print(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

start_time=time.time()

model = LOF(n_neighbors=7,contamination=0.1)
#model = ECOD()

model.fit(X_scaled)

anomaly_scores = model.decision_scores_

threshold = 1000.00

anomalies = df[anomaly_scores > threshold]
print("Detected anomalies (Transaction Amount > {}:".format(threshold))
print(anomalies)

runtime = round(time.time() - start_time, 3)

print(runtime)