import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn, model_rf, model_nb, model_cb, model_svm, model_xgboost
from utils.unsupervised_learning import model_lof, model_pca, model_iforest, model_ecod, model_copod, model_kmeans
import matplotlib.pyplot as plt
from math import sqrt

datasets = [50000, 150000, 350000, 550000, 750000, 1000000]

roc_auc = {
    'KNN': [],
    'Random Forest': [],
    'XGBoost': [],
    'SVM': [],
    'Naive Bayes': [],
    'CATboost': [],
    'LOF': [],
    'PCA': [],
    'Isolation Forest': [],
    'K-Means': [],
    'COPOD': [],
    'ECOD': []
}

f1_scores = {
    'KNN': [],
    'Random Forest': [],
    'XGBoost': [],
    'SVM': [],
    'Naive Bayes': [],
    'CATboost': [],
    'LOF': [],
    'PCA': [],
    'Isolation Forest': [],
    'K-Means': [],
    'COPOD': [],
    'ECOD': []
}

runtimes = {
    'KNN': [],
    'Random Forest': [],
    'XGBoost': [],
    'SVM': [],
    'Naive Bayes': [],
    'CATboost': [],
    'LOF': [],
    'PCA': [],
    'Isolation Forest': [],
    'K-Means': [],
    'COPOD': [],
    'ECOD': []
}

for dataset in datasets:

    X, y = make_classification(n_samples=dataset, n_features=15, n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    '''
    ===============================
    SUPERVISED LEARNING ALGORITHMS
    ===============================
    
    Testing the scalability of the supervised models with default parameters for each model.
    '''
    # MODEL K-NEAREST NEIGHBORS (KNN)
    k_value = 10
    # Evaluate the KNN model using the best parameters
    roc_auc_knn, f1_score_knn, runtime_knn = model_knn(X_train, X_test, y_train, y_test, k_value)

    roc_auc['KNN'].append(roc_auc_knn)
    f1_scores['KNN'].append(f1_score_knn)
    runtimes['KNN'].append(runtime_knn)


    # MODEL RANDOM FOREST (RF)
    rf_value = 100
    # Evaluate the KNN model using the best parameters
    roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_value)

    roc_auc['Random Forest'].append(roc_auc_rf)
    f1_scores['Random Forest'].append(f1_score_rf)
    runtimes['Random Forest'].append(runtime_rf)

    # MODEL XGBOOST
    xgboost_value = 100
    xgboost_depth = 3

    # Evaluate the XGBoost model using the best parameters
    roc_auc_xgboost, f1_score_xgboost, runtime_xgboost = model_xgboost(X_train, X_test, y_train, y_test, xgboost_value,
                                                                       xgboost_depth)

    roc_auc['XGBoost'].append(roc_auc_xgboost)
    f1_scores['XGBoost'].append(f1_score_xgboost)
    runtimes['XGBoost'].append(runtime_xgboost)

    # MODEL SUPPORT VECTOR MACHINE (SVM)
    # Evaluate the SVM model
    roc_auc_svm, f1_score_svm, runtime_svm = model_svm(X_train, X_test, y_train, y_test)

    roc_auc['SVM'].append(roc_auc_svm)
    f1_scores['SVM'].append(f1_score_svm)
    runtimes['SVM'].append(runtime_svm)

    # MODEL NAIVE BAYES (NB)
    # Evaluate the Naive Bayes model
    roc_auc_nb, f1_score_nb, runtime_nb = model_nb(X_train, X_test, y_train, y_test)

    roc_auc['Naive Bayes'].append(roc_auc_nb)
    f1_scores['Naive Bayes'].append(f1_score_nb)
    runtimes['Naive Bayes'].append(runtime_nb)

    # MODEL CATBOOST (CB)
    cb_iterations = 100
    cb_learning_rate = 0.1
    cb_depth = 3
    # Evaluate the CatBoost model
    roc_auc_cb, f1_score_cb, runtime_cb = model_cb(X_train, X_test, y_train, y_test, cb_iterations, cb_learning_rate,
                                                   cb_depth)

    roc_auc['CATboost'].append(roc_auc_cb)
    f1_scores['CATboost'].append(f1_score_cb)
    runtimes['CATboost'].append(runtime_cb)

    '''
    #================================
    #UNSUPERVISED LEARNING ALGORITHMS
    #================================
    
    #Testing the scalability of the unsupervised models with default parameters for each model.

    '''

    # MODEL LOCAL OUTLIER FACTOR (LOF)
    k_lof = 20

    # Evaluate the LOF model
    roc_auc_lof, f1_score_lof, runtime_lof = model_lof(X, y, k_lof)

    roc_auc['LOF'].append(roc_auc_lof)
    f1_scores['LOF'].append(f1_score_lof)
    runtimes['LOF'].append(runtime_lof)

    # MODEL PRINCIPAL COMPONENT ANALYSIS (PCA)
    # Evaluate the PCA model
    roc_auc_pca, f1_score_pca, runtime_pca = model_pca(X, y)

    roc_auc['PCA'].append(roc_auc_pca)
    f1_scores['PCA'].append(f1_score_pca)
    runtimes['PCA'].append(runtime_pca)

    # MODEL ISOLATION FOREST (IF)
    # Tune the Isolation Forest model to get the best hyperparameters
    if_value = 100

    # Evaluate the IF model
    roc_auc_if, f1_score_if, runtime_if = model_iforest(X, y, if_value)

    roc_auc['Isolation Forest'].append(roc_auc_if)
    f1_scores['Isolation Forest'].append(f1_score_if)
    runtimes['Isolation Forest'].append(runtime_if)

    # MODEL K-Means clustering
    # Evaluate the K-means model
    roc_auc_kmeans, f1_score_kmeans, runtime_kmeans = model_kmeans(X, y, 8) #Using default value of k=8

    roc_auc['K-Means'].append(roc_auc_kmeans)
    f1_scores['K-Means'].append(f1_score_kmeans)
    runtimes['K-Means'].append(runtime_kmeans)

    # MODEL COPULA BASED OUTLIER DETECTION (COPOD)
    # Evaluate the COPOD model
    roc_auc_copod, f1_score_copod, runtime_copod = model_copod(X, y)

    roc_auc['COPOD'].append(roc_auc_copod)
    f1_scores['COPOD'].append(f1_score_copod)
    runtimes['COPOD'].append(runtime_copod)

    # MODEL EMPIRICAL CUMULATIVE DISTRIBUTION BASED OUTLIER DETECTION (ECOD)
    # Evaluate the ECOD model
    roc_auc_ecod, f1_score_ecod, runtime_ecod = model_ecod(X, y)

    roc_auc['ECOD'].append(roc_auc_ecod)
    f1_scores['ECOD'].append(f1_score_ecod)
    runtimes['ECOD'].append(runtime_ecod)

# In this section, the results of the evaluation are saved and used to create the necessary visualizations.

# Save the metrics to a CSV file
pd.DataFrame(roc_auc).to_csv('Scalability_test(ROC_AUC).csv', index=False)
pd.DataFrame(f1_scores).to_csv('Scalability_test(F1_scores).csv', index=False)
pd.DataFrame(runtimes).to_csv('Scalability_test(Runtime).csv', index=False)

# Visualizing the results
#Visualize the ROC_AUC scores
plt.figure(figsize=(10, 6))
for model, scores in roc_auc.items():
    plt.plot(datasets, scores, marker='o', linestyle='--', label=model)
plt.xlabel('Dataset size')
plt.ylabel('ROC AUC Scores')
plt.legend(loc='best')
plt.savefig('./Scalability_test(ROC_AUC).png')
plt.show()

# Visualize the F1 scores
plt.figure(figsize=(10, 6))
for model, scores in f1_scores.items():
    plt.plot(datasets, scores, marker='o', linestyle='--', label=model)
plt.xlabel('Dataset size')
plt.ylabel('F1 Scores')
plt.legend(loc='best')
plt.savefig('./Scalability_test(F1_scores).png')
plt.show()

# Visualize the Runtimes scores
plt.figure(figsize=(10, 6))
for model, scores in runtimes.items():
    plt.plot(datasets, scores, marker='o', linestyle='--', label=model)
plt.xlabel('Dataset size')
plt.ylabel('Runtime Scores')
plt.legend(loc='best')
plt.savefig('./Scalability_test(Runtime).png')
plt.show()

