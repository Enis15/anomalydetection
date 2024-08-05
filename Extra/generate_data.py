import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize

from utils.supervised_learning import model_knn, model_rf, model_nb, model_cb, model_svm, model_xgboost
from utils.unsupervised_learning import model_lof, model_pca, model_iforest, model_ecod, model_copod, model_dbscan
import matplotlib.pyplot as plt
# Import the logger function
from utils.logger import logger

# Initialize the logger
_logger = logger(__name__)


datasets = [50000, 150000, 350000, 550000, 750000, 1000000]

roc_auc = {
    'DBSCAN': [],
}

f1_scores = {
    'DBSCAN': [],

}
runtimes = {
    'DBSCAN': [],

}

for dataset in datasets:

    X, y = make_classification(n_samples=dataset, n_features=25, n_classes=2, random_state=42)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = normalize(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    '''
    ===============================
    SUPERVISED LEARNING ALGORITHMS
    ===============================

    Testing the scalability of the supervised models with default parameters for each model.
    '''
    try:
        # MODEL DBSCAN clustering
        _logger.info('Starting DBSCAN Evaluation')
        # Evaluate the DBSCAN model
        roc_auc_dbscan, f1_score_dbscan, runtime_dbscan = model_dbscan(X, y, eps=0.8, min_samples=150)
        _logger.info(f'DBSCAN Evaluation: ROC AUC Score={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, Runtime={runtime_dbscan}')
        roc_auc['DBSCAN'].append(roc_auc_dbscan)
        f1_scores['DBSCAN'].append(f1_score_dbscan)
        runtimes['DBSCAN'].append(runtime_dbscan)
    except Exception as e:
        _logger.error(f'Error evaluating DBSCAN model:{e}')




'''
plt.figure(figsize=(10, 6))
#Plot the ROC_AUC score
#plt.subplot(1, 2, 1)
for model, scores in roc_auc.items():
    plt.plot(datasets, scores, marker='o', linestyle='-', label=model)
plt.xlabel('Dataset size')
plt.ylabel('Scores')
plt.title('Dataset size vs ROC_AUC Metrics')
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(10, 6))
#plt.subplot(1, 2, 2)
for model, scores in f1_scores.items():
    plt.plot(datasets, scores, marker='o', linestyle='--', label=model)
plt.xlabel('Dataset size')
plt.ylabel('f1_Scores')
plt.title('Dataset size vs F1 Metrics')
plt.legend(loc='best')
plt.show()

plt.figure(figsize=(10, 6))
for model, scores in runtimes.items():
    plt.plot(datasets, scores, marker='o', linestyle='-', label=model)
plt.xlabel('Dataset size')
plt.ylabel('Runtime')
plt.title('Dataset size vs runtime')
plt.legend(loc='best')
plt.show()
'''
