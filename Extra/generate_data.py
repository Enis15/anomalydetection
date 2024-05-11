'''
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn
X, y = make_classification(n_samples=10000, n_classes=2 ,n_features=15, random_state=5)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
knn = model_knn(X_train, X_test, y_train, y_test, 5)
'''
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn, model_rf, model_nb, model_cb, model_svm, model_xgboost
from utils.unsupervised_learning import model_lof, model_pca, model_iforest, model_ecod, model_copod, model_cblof
import matplotlib.pyplot as plt
from math import sqrt

datasets = [50, 150, 350, 550, 750, 1000]

roc_auc = {
    'CBLOF': [],
    'Random Forest': [],
    'PCA': []
}
f1_scores = {
    'CBLOF': [],
    'Random Forest': [],
    'PCA': []
}
runtimes = {
    'CBLOF': [],
    'Random Forest': [],
    'PCA': []
}

for i in datasets:
    X, y = make_classification(n_samples=i, n_features=15, n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    '''
    ===============================
    SUPERVISED LEARNING ALGORITHMS
    ===============================

    Testing the scalability of the supervised models with default parameters for each model.
    '''
    # MODEL K-NEAREST NEIGHBORS (CBLOF)
    k_value = 8
    # Evaluate the CBLOF model using the best parameters
    roc_auc_cblof, f1_score_cblof, runtime_cblof = model_cblof(X, y)

    roc_auc['CBLOF'].append(roc_auc_cblof)
    f1_scores['CBLOF'].append(f1_score_cblof)
    runtimes['CBLOF'].append(runtime_cblof)

    # MODEL RANDOM FOREST (RF)
    rf_value = 10
    # Evaluate the KNN model using the best parameters
    roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_value)

    roc_auc['Random Forest'].append(roc_auc_rf)
    f1_scores['Random Forest'].append(f1_score_rf)
    runtimes['Random Forest'].append(runtime_rf)
    # Evaluate the PCA model
    roc_auc_pca, f1_score_pca, runtime_pca = model_pca(X, y)

    roc_auc['PCA'].append(roc_auc_pca)
    f1_scores['PCA'].append(f1_score_pca)
    runtimes['PCA'].append(runtime_pca)


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
