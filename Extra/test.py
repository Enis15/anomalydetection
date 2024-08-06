from sklearn.datasets import make_classification
from utils.unsupervised_learning import model_iforest, model_ecod
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score


X, y = make_classification(n_samples=1000000, n_features=15, n_classes=2, random_state=42)

# Setting the fold splits for unsupervised learning models
scorer = {'f1_score': make_scorer(f1_score), 'roc_auc': make_scorer(roc_auc_score)} # Metrics for cross validation performance
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Fold splits

forest_estimator = 1000
#roc_auc_if, f1_score_if, runtime_if = model_iforest(X, y, forest_estimator)

#print(roc_auc_if, f1_score_if, runtime_if)

roc_auc_ecod, f1_score_ecod, runtime_ecod = model_ecod(X, y)