import pandas as pd
from sklearn.model_selection import train_test_split
from models.ds1_creditcard import X_train, X_test, y_train, y_test
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import f1_score, accuracy_score
from utils.paramet_tune import Catboost_tuner, LOF_tuner, Kmeans_tuner, RandomForest_tuner
from utils.unsupervised_learning import model_lof, model_kmeans
from utils.supervised_learning import model_cb, model_rf, model_knn, model_nb, model_svm
from utils.paramet_tune import paramet_tune
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from adjustText import adjust_text
from utils.logger import logger

_logger = logger(__name__)

if __name__ == '__main__':
    # Ensure compatibility
    freeze_support()

    # DataFrame to store the evaluation metrics
    metrics = []

    # Function to append results to metrics list
    def append_metrics(modelname, estimator, roc_auc, f1_score, runtime):
        metrics.append({
            'Model': modelname,
            'Estimator': estimator,
            'ROC_AUC_Score': roc_auc,
            'F1 Score': f1_score,
            'Runtime': runtime
        })

    _logger.info('Starting model evaluation')

    try:
        # MODEL SUPPORT VECTOR MACHINE (SVM)
        _logger.info('Evaluating SVM model')
        roc_auc_svm, f1_score_svm, runtime_svm = model_svm(X_train, X_test, y_train, y_test)
        append_metrics('SVM', None, roc_auc_svm, f1_score_svm, runtime_svm)
        _logger.info(f'SVM Evaluation: ROC AUC={roc_auc_svm}, F1 Score={f1_score_svm}, Runtime={runtime_svm}')
    except Exception as e:
        _logger.error(f'Error evaluating SVM model: {e}')

    try:
        # MODEL NAIVE BAYES (NB)
        _logger.info('Evaluating Naive Bayes model')
        roc_auc_nb, f1_score_nb, runtime_nb = model_nb(X_train, 5, y_train, y_test)
        append_metrics('Naive Bayes', None, roc_auc_nb, f1_score_nb, runtime_nb)
        _logger.info(f'Naive Bayes Evaluation: ROC AUC={roc_auc_nb}, F1 Score={f1_score_nb}, Runtime={runtime_nb}')
    except Exception as e:
        _logger.error(f'Error evaluating Naive Bayes model: {e}')









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



if __name__ == '__main__':

    freeze_support()
    metrics = []

    def append_metrics(model_name, estimator, roc_auc, f1_score, runtime):
        metrics.append({
            'Model': model_name,
            'Estimator': estimator,
            'ROC_AUC': roc_auc,
            'F1_Score': f1_score,
            'Runtime': runtime
        })

    rf_tuner = RandomForest_tuner(X_train, X_test, y_train, y_test)
    best_rf_model = rf_tuner.tune_model()

    rf_value = int(best_rf_model['n_estimators'])
    rf_depth = int(best_rf_model['max_depth'])
    roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_value, max_depth=rf_depth)
    append_metrics('Random Forest', rf_value, roc_auc_rf, f1_score_rf, runtime_rf)
    '''