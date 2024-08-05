from sklearn.datasets import make_classification
from utils.paramet_tune import LOF_tuner


X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, random_state=42)

# catboost_tunne = Catboost_tuner(X, y)
# best_catboost = catboost_tunne.tune_model()
#
# cb_iterations = int(best_catboost['iterations'])
# cb_learning_rate = best_catboost['learning_rate']
# cb_depth = int(best_catboost['depth'])

lof_tune = LOF_tuner(X, y)
best_lof = lof_tune.tune_model()

# scorer = {'f1_score':make_scorer(f1_score), 'roc_auc':make_scorer(roc_auc_score)}
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(best_lof)