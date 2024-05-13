
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import f1_score, accuracy_score
from utils.paramet_tune import Catboost_tune, LOF_tune, Kmeans_tune
from utils.unsupervised_learning import model_lof, model_kmeans
from utils.supervised_learning import model_cb


df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

#Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

'''
def objective(params):
    model = CatBoostClassifier(
        iterations=int(params['iterations']),
        learning_rate=params['learning_rate'],
        depth=int(params['depth']),
        loss_function='Logloss',
        verbose=False
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_score(y_test, y_pred)

    return {'loss': -1, 'status': STATUS_OK}

space = {
    'iterations': hp.quniform('iterations', 100, 500, 10),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'depth': hp.quniform('depth', 4, 10, 1),
}

trials = Trials()

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
print('Best parameters:', best)


#tuner = Catboost_tune(X_train, X_test, y_train, y_test)
#tuner.tune_model()

tuner = Catboost_tune(X_train, X_test, y_train, y_test)
best_hyperparameters = tuner.tune_model()
print(best_hyperparameters)

best_iterations = int(best_hyperparameters['iterations'])
best_learning_rate = best_hyperparameters['learning_rate']
best_depth = int(best_hyperparameters['depth'])

model = model_cb(X_train, X_test, y_train, y_test, best_iterations, best_learning_rate, best_depth)

'''

k_means_tuner = Kmeans_tune(X_train)
k_clusters = k_means_tuner.tune_model()
# Evaluate the K-Means model
roc_auc_kmeans, f1_score_kmeans, runtime_kmeans = model_kmeans(X, y, k_clusters)