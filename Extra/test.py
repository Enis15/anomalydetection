from sklearn.datasets import make_classification
from utils.unsupervised_learning import model_iforest, model_dbscan, model_copod, model_ecod, model_pca, model_lof
from utils.supervised_learning import model_nb
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd

# Setting the fold splits for unsupervised learning models
scorer = {'f1_score': make_scorer(f1_score), 'roc_auc': make_scorer(roc_auc_score)} # Metrics for cross validation performance
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Fold splits


df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

# Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

scaler = StandardScaler()
X = scaler.fit_transform(X) # Standardize the data

X = normalize(X) # Normalize the data

roc_auc, f1_score, runtime = model_dbscan(X, y, 20, 15)