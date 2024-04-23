import pandas as pd
from pyod.models.lof import LOF
from sklearn.metrics import roc_auc_score, f1_score
import time
from utils.unsupervisedl_learning import model_lof

df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')
X = df.drop(['Class'], axis=1)
y = df['Class'].values

model = model_lof(X, y, 3)
