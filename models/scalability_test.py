import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import the logger function
from utils.logger import logger

# Supervised learning models
from utils.supervised_learning import model_knn, model_rf, model_nb, model_cb, model_svm, model_xgboost
# Unsupervised learning models
from utils.unsupervised_learning import model_lof, model_pca, model_iforest, model_ecod, model_copod, model_dbscan

# Initialize the logger
_logger = logger(__name__)

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
    'DBSCAN': [],
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
    'DBSCAN': [],
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
    'DBSCAN': [],
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
    try:
        # MODEL K-NEAREST NEIGHBORS (KNN)
        _logger.info('Starting KNN Evaluation')
        k_value = 10
        # Evaluate the KNN model using the best parameters
        roc_auc_knn, f1_score_knn, runtime_knn = model_knn(X_train, X_test, y_train, y_test, k_value)
        _logger.info(f'KNN Evaluation: ROC AUC SCORE={roc_auc_knn}, F1 SCORE={f1_score_knn}, Runtime={runtime_knn}')
        roc_auc['KNN'].append(roc_auc_knn)
        f1_scores['KNN'].append(f1_score_knn)
        runtimes['KNN'].append(runtime_knn)
    except Exception as e:
        _logger.error(f'Error evaluating KNN model:{e}')

    try:
        # MODEL RANDOM FOREST (RF)
        _logger.info('Starting Random Forest Classifier Evaluation')
        rf_value = 100
        # Evaluate the KNN model using the best parameters
        roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_value)
        _logger.info(f'Random Forest Classifier Evaluation: ROC AUC SCORE={roc_auc_rf}, F1 SCORE={f1_score_rf}, Runtime={runtime_rf}')
        roc_auc['Random Forest'].append(roc_auc_rf)
        f1_scores['Random Forest'].append(f1_score_rf)
        runtimes['Random Forest'].append(runtime_rf)
    except Exception as e:
        _logger.error(f'Error evaluating Random Forest Classifier model:{e}')

    try:
        # MODEL XGBOOST
        _logger.info('Starting XGBoost Classifier Evaluation')
        xgboost_value = 100
        xgboost_depth = 3
        # Evaluate the XGBoost model using the best parameters
        roc_auc_xgboost, f1_score_xgboost, runtime_xgboost = model_xgboost(X_train, X_test, y_train, y_test, xgboost_value,
                                                                           xgboost_depth)
        _logger.info(f'XGBoost Evaluation: ROC AUC Score={roc_auc_xgboost}, F1 Score={f1_score_xgboost}, Runtime={runtime_xgboost}')
        roc_auc['XGBoost'].append(roc_auc_xgboost)
        f1_scores['XGBoost'].append(f1_score_xgboost)
        runtimes['XGBoost'].append(runtime_xgboost)
    except Exception as e:
        _logger.error(f'Error evaluating XGBoost Classifier model:{e}')

    try:
        # MODEL SUPPORT VECTOR MACHINE (SVM)
        _logger.info('Starting SVM Classifier Evaluation')
        # Evaluate the SVM model
        roc_auc_svm, f1_score_svm, runtime_svm = model_svm(X_train, X_test, y_train, y_test)
        _logger.info(f'SVM Evaluation: ROC AUC Score={roc_auc_svm}, F1 Score={f1_score_svm}, Runtime={runtime_svm}')
        roc_auc['SVM'].append(roc_auc_svm)
        f1_scores['SVM'].append(f1_score_svm)
        runtimes['SVM'].append(runtime_svm)
    except Exception as e:
        _logger.error(f'Error evaluating SVM model:{e}')

    try:
    # MODEL NAIVE BAYES (NB)
        _logger.info('Starting Naive Bayes Classifier Evaluation')
        # Evaluate the Naive Bayes model
        roc_auc_nb, f1_score_nb, runtime_nb = model_nb(X_train, X_test, y_train, y_test)
        _logger.info(f'Naive Bayes Evaluation: ROC AUC Score={roc_auc_nb}, F1 Score={f1_score_nb}, Runtime={runtime_nb}')
        roc_auc['Naive Bayes'].append(roc_auc_nb)
        f1_scores['Naive Bayes'].append(f1_score_nb)
        runtimes['Naive Bayes'].append(runtime_nb)
    except Exception as e:
        _logger.error(f'Error evaluating Naive Bayes model:{e}')

    try:
        # MODEL CATBOOST (CB)
        _logger.info('Starting CatBoost Classifier Evaluation')
        cb_iterations = 100
        cb_learning_rate = 0.1
        cb_depth = 3
        # Evaluate the CatBoost model
        roc_auc_cb, f1_score_cb, runtime_cb = model_cb(X_train, X_test, y_train, y_test, cb_iterations, cb_learning_rate,
                                                       cb_depth)
        _logger.info(f'CatBoost Evaluation: ROC AUC Score= {roc_auc_cb}, F1 Score= {f1_score_cb}, Runtime= {runtime_cb}')
        roc_auc['CATboost'].append(roc_auc_cb)
        f1_scores['CATboost'].append(f1_score_cb)
        runtimes['CATboost'].append(runtime_cb)
    except Exception as e:
        _logger.error(f'Error evaluating CatBoost model:{e}')

    '''
    #================================
    #UNSUPERVISED LEARNING ALGORITHMS
    #================================
    
    #Testing the scalability of the unsupervised models with default parameters for each model.

    '''
    try:
        # MODEL LOCAL OUTLIER FACTOR (LOF)
        _logger.info('Starting LOF Evaluation')
        k_lof = 20
        # Evaluate the LOF model
        roc_auc_lof, f1_score_lof, runtime_lof = model_lof(X, y, k_lof)
        _logger.info(f'LOF Evaluation: ROC AUC Score={roc_auc_lof}, F1 Score={f1_score_lof}, Runtime={runtime_lof}')
        roc_auc['LOF'].append(roc_auc_lof)
        f1_scores['LOF'].append(f1_score_lof)
        runtimes['LOF'].append(runtime_lof)
    except Exception as e:
        _logger.error(f'Error evaluating LOF model:{e}')

    try:
        # MODEL PRINCIPAL COMPONENT ANALYSIS (PCA)
        _logger.info('Starting PCA Evaluation')
        # Evaluate the PCA model
        roc_auc_pca, f1_score_pca, runtime_pca = model_pca(X, y)
        _logger.info(f'PCA Evaluation: ROC AUC Score={roc_auc_pca}, F1 Score={f1_score_pca}, Runtime={runtime_pca}')
        roc_auc['PCA'].append(roc_auc_pca)
        f1_scores['PCA'].append(f1_score_pca)
        runtimes['PCA'].append(runtime_pca)
    except Exception as e:
        _logger.error(f'Error evaluating PCA model:{e}')

    try:
        # MODEL ISOLATION FOREST (IF)
        _logger.info('Starting Isolation Forest Evaluation')
        if_value = 100
        # Evaluate the IF model
        roc_auc_if, f1_score_if, runtime_if = model_iforest(X, y, if_value)
        _logger.info(f'Isolation Forest Evaluation: ROC AUC Score={roc_auc_if}, F1 Score={f1_score_if}, Runtime={runtime_if}')
        roc_auc['Isolation Forest'].append(roc_auc_if)
        f1_scores['Isolation Forest'].append(f1_score_if)
        runtimes['Isolation Forest'].append(runtime_if)
    except Exception as e:
        _logger.error(f'Error evaluating Isolation Forest model:{e}')

    try:
        # MODEL DBSCAN clustering
        _logger.info('Starting DBSCAN Evaluation')
        # Evaluate the K-means model
        roc_auc_dbscan, f1_score_dbscan, runtime_dbscan = model_dbscan(X, y, eps=0.5, min_samples=5) #Using default value of k=8
        _logger.info(f'DBSCAN Evaluation: ROC AUC Score={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, Runtime={runtime_dbscan}')
        roc_auc['DBSCAN'].append(roc_auc_dbscan)
        f1_scores['DBSCAN'].append(f1_score_dbscan)
        runtimes['DBSCAN'].append(runtime_dbscan)
    except Exception as e:
        _logger.error(f'Error evaluating DBSCAN model:{e}')

    try:
        # MODEL COPULA BASED OUTLIER DETECTION (COPOD)
        _logger.info('Starting COPOD Evaluation')
        # Evaluate the COPOD model
        roc_auc_copod, f1_score_copod, runtime_copod = model_copod(X, y)
        _logger.info(f'COPOD Evaluation: ROC AUC Score={roc_auc_copod}, F1 Score={f1_score_copod}, Runtime={runtime_copod}')
        roc_auc['COPOD'].append(roc_auc_copod)
        f1_scores['COPOD'].append(f1_score_copod)
        runtimes['COPOD'].append(runtime_copod)
    except Exception as e:
        _logger.error(f'Error evaluating COPOD model:{e}')

    try:
        # MODEL EMPIRICAL CUMULATIVE DISTRIBUTION BASED OUTLIER DETECTION (ECOD)
        _logger.info('Starting ECOD Evaluation')
        # Evaluate the ECOD model
        roc_auc_ecod, f1_score_ecod, runtime_ecod = model_ecod(X, y)
        _logger.info(f'ECOD Evaluation: ROC AUC Score={roc_auc_ecod}, F1 Score={f1_score_ecod}, Runtime={runtime_ecod}')
        roc_auc['ECOD'].append(roc_auc_ecod)
        f1_scores['ECOD'].append(f1_score_ecod)
        runtimes['ECOD'].append(runtime_ecod)
    except Exception as e:
        _logger.error(f'Error evaluating ECODs model:{e}')

# In this section, the results of the evaluation are saved and used to create the necessary visualizations.

# Save the metrics to a CSV file
pd.DataFrame(roc_auc).to_csv('../results/Scalability_test(ROC_AUC).csv', index=False)
pd.DataFrame(f1_scores).to_csv('../results/Scalability_test(F1_scores).csv', index=False)
pd.DataFrame(runtimes).to_csv('../results/Scalability_test(Runtime).csv', index=False)

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

# try:
#     # MODEL K-Means clustering
#     _logger.info('Starting K-Means Evaluation')
#     # Evaluate the K-means model
#     roc_auc_kmeans, f1_score_kmeans, runtime_kmeans = model_kmeans(X, y, 8)  # Using default value of k=8
#     _logger.info(
#         f'K Means Evaluation: ROC AUC Score={roc_auc_kmeans}, F1 Score={f1_score_kmeans}, Runtime={runtime_kmeans}')
#     roc_auc['K-Means'].append(roc_auc_kmeans)
#     f1_scores['K-Means'].append(f1_score_kmeans)
#     runtimes['K-Means'].append(runtime_kmeans)
# except Exception as e:
#     _logger.error(f'Error evaluating K Means model:{e}')