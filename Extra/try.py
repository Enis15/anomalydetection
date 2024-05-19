import pandas as pd
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import f1_score, accuracy_score
from utils.paramet_tune import Catboost_tuner, LOF_tuner, Kmeans_tuner, RandomForest_tuner
from utils.unsupervised_learning import model_lof, model_kmeans
from utils.supervised_learning import model_cb, model_rf
from utils.paramet_tune import paramet_tune
from multiprocessing import freeze_support
import matplotlib.pyplot as plt


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

metrics_df = pd.DataFrame({
        'Model': ['KNN', 'Random Forest Classifier', 'XGBoost', 'SVM', 'Naive Bayes', 'CATBoost', 'LOF', 'PCA', 'Isolation_Forest', 'K-means', 'COPOD', 'ECOD'],
        'Estimator': [8, 190, 3400, None, None, 330, 13, None, 66, 2, None, None],
        'ROC_AUC_Score': [0.504, 0.904, 0.908, 0.5, 0.828, 0.923, 0.638, 0.902, 0.889, 0.433, 0.887, 0.893],
        'F1 Score': [0.998, 1.0, 1.0, 0.998, 0.995, 1.0, 0.991, 0.946, 0.946, 0.697, 0.946, 0.946],
        'Runtime': [16.894, 62.63, 0.901, 9.378, 0.074, 5.362, 289.7, 0.705, 2.165, 0.389, 0.389, 14.229]})

# Save the metrics to a CSV file
metrics_df.to_csv('./Metrics(DS1).csv', index=False)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(metrics_df['Runtime'], metrics_df['ROC_AUC_Score'], color='blue')
for i, txt in enumerate(metrics_df['Model']):
    plt.annotate(txt, (metrics_df['Runtime'][i], metrics_df['ROC_AUC_Score'][i]))
plt.xlabel('Runtime')
plt.ylabel('ROC AUC')
plt.title('ROC AUC vs Runtime comparison')
plt.savefig('./ROC_AUC_vs_Runtime(2).png', bbox_inches='tight')
plt.show()

