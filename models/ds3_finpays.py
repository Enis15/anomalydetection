import os
import pandas as pd
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from multiprocessing import freeze_support
from adjustText import adjust_text
import matplotlib.pyplot as plt

# Import the logger function
from utils.logger import logger

# Supervised learning models
from utils.supervised_learning import model_knn, model_xgboost, model_svm, model_cb, model_nb, model_rf
# Unsupervised learning models
from utils.unsupervised_learning import model_lof, model_iforest, model_ecod, model_pca, model_dbscan, model_copod
# Hyperparameter tuning functions
from utils.paramet_tune import IsolationForest_tuner, KNN_tuner, XGBoost_tuner, Catboost_tuner, LOF_tuner, RandomForest_tuner, DBSCAN_tuner

# Initialize the logger
_logger = logger(__name__)

'''
Dataset 3: Bank Payment dataset with over 600.000 records and 10 features.
'''

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/fin_paysys/finpays.csv')

# Encoding categorical values with numerical variables
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

# Determining the X and y values
X = df.drop('fraud', axis=1)
y = df['fraud'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardize the data

# Setting the fold splits for supervised learning models
scorer = {
    'f1_score': make_scorer(f1_score, average='weighted'),
    'roc_auc': make_scorer(roc_auc_score, average='weighted')}  # Metrics for cross validation performance

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Fold splits

if __name__ == '__main__':
    # Ensure compatability
    freeze_support()

    # DataFrame to store the evaluation metrics
    metrics = []

    metrics_unsupervised = []

    # Function to append results to metrics list
    def append_metrics(modelname, estimator, roc_auc, f1_score, runtime):
        metrics.append({
            'Model': modelname,
            'Estimator': estimator,
            'ROC_AUC_Score': roc_auc,
            'F1_Score': f1_score,
            'Runtime': runtime
        })

    def unsupervised_metrics(modelname, estimator, roc_auc, f1_score, precision, recall, accuracy, runtime):
        metrics_unsupervised.append({
            'Model': modelname,
            'Estimator': estimator,
            'ROC_AUC_Score': roc_auc,
            'F1_Score': f1_score,
            'Precision': precision,
            'Recall': recall,
            'Accuracy': accuracy,
            'Runtime': runtime
        })


    '''
    ===============================
    SUPERVISED LEARNING ALGORITHMS
    ===============================

    This section of the code evaluates the performance of supervised learning algorithms, incl. KNN(K-Nearest Neighbor), 
    Random Forest Classifier, XGBoost(Extreme Gradient Boosting), SVM(Support Vector Machine), Naive Bayes and 
    CATBoost(Categorical Boosting). Some of the algorithms have been fine-tuned using the hyperopt library, 
    while the others use default parameters provided by sklearn library.
    '''
    try:
        # MODEL K-NEAREST NEIGHBORS (KNN)
        _logger.info('Starting KNN Evaluation')
        # Tune the KNN model to get the best hyperparameters
        knn_tuner = KNN_tuner(X_scaled, y)
        best_knn = knn_tuner.tune_model()
        _logger.info(f'Best K-Nearest Neighbor Model: {best_knn}')
        k_value = int(best_knn['n_neighbors'])  # Save the value of k
        # Evaluate the KNN model using the best parameters
        roc_auc_knn, f1_score_knn, runtime_knn = model_knn(X_scaled, y, k_value, scorer, kf)
        append_metrics('KNN', k_value, roc_auc_knn, f1_score_knn, runtime_knn)
        _logger.info(f'KNN Evaluation: ROC AUC SCORE={roc_auc_knn}, F1 SCORE={f1_score_knn}, Runtime={runtime_knn}')
    except Exception as e:
        _logger.error(f'Error evaluating KNN model:{e}')

    try:
        # MODEL RANDOM FOREST (RF)
        _logger.info('Starting Random Forest Classifier Evaluation')
        # Tune the Random Forest model to get the best hyperparameters
        rf_tuner = RandomForest_tuner(X, y)
        best_rf_model = rf_tuner.tune_model()
        _logger.info(f'Best Random Forest Model: {best_rf_model}')
        rf_estimator = int(best_rf_model['n_estimators'])
        rf_depth = int(best_rf_model['max_depth'])
        # Evaluate the Random Forest Classifier using the best parameters
        roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X, y, rf_estimator, rf_depth, scorer, kf)
        append_metrics('Random Forest Classifier', rf_estimator, roc_auc_rf, f1_score_rf, runtime_rf)
        _logger.info(
            f'Random Forest Classifier Evaluation: ROC AUC SCORE={roc_auc_rf}, F1 SCORE={f1_score_rf}, Runtime={runtime_rf}')
    except Exception as e:
        _logger.error(f'Error evaluating Random Forest Classifier model:{e}')

    try:
        # MODEL XGBOOST
        _logger.info('Starting XGBoost Classifier Evaluation')
        # Tune the XGBOOST model to get the best hyperparameters
        xgboost_tuner = XGBoost_tuner(X, y)
        best_xgboost_model = xgboost_tuner.tune_model()
        _logger.info(f'Best XGBoost Model: {best_xgboost_model}')
        xgboost_value = int(best_xgboost_model['n_estimators'])  # Save the value of n_estimators
        xgboost_depth = int(best_xgboost_model['max_depth'])  # Save the value of max_depth
        xgboost_learn_rate = best_xgboost_model['n_estimators']

        # Evaluate the XGBoost model using the best parameters
        roc_auc_xgboost, f1_score_xgboost, runtime_xgboost = model_xgboost(X, y, xgboost_value, xgboost_depth, xgboost_learn_rate, scorer, kf)
        append_metrics('XGBoost', xgboost_value, roc_auc_xgboost, f1_score_xgboost, runtime_xgboost)
        _logger.info(
            f'XGBoost Evaluation: ROC AUC Score={roc_auc_xgboost}, F1 Score={f1_score_xgboost}, Runtime={runtime_xgboost}')
    except Exception as e:
        _logger.error(f'Error evaluating XGBoost Classifier model:{e}')

    try:
        # MODEL SUPPORT VECTOR MACHINE (SVM)
        _logger.info('Starting SVM Classifier Evaluation')
        # Evaluate the SVM model
        roc_auc_svm, f1_score_svm, runtime_svm = model_svm(X_scaled, y, scorer, kf)
        append_metrics('SVM', None, roc_auc_svm, f1_score_svm, runtime_svm)
        _logger.info(f'SVM Evaluation: ROC AUC Score={roc_auc_svm}, F1 Score={f1_score_svm}, Runtime={runtime_svm}')
    except Exception as e:
        _logger.error(f'Error evaluating SVM model:{e}')

    try:
        # MODEL NAIVE BAYES (NB)
        _logger.info('Starting Naive Bayes Classifier Evaluation')
        # Evaluate the Naive Bayes model
        roc_auc_nb, f1_score_nb, runtime_nb = model_nb(X, y, scorer, kf)
        append_metrics('Naive Bayes', None, roc_auc_nb, f1_score_nb, runtime_nb)
        _logger.info(
            f'Naive Bayes Evaluation: ROC AUC Score={roc_auc_nb}, F1 Score={f1_score_nb}, Runtime={runtime_nb}')
    except Exception as e:
        _logger.error(f'Error evaluating Naive Bayes model:{e}')

    try:
        # MODEL CATBOOST (CB)
        _logger.info('Starting CatBoost Classifier Evaluation')
        # Tune the CatBoost model to get the best hyperparameters
        catboost_tuner = Catboost_tuner(X, y)
        best_catboost = catboost_tuner.tune_model()
        _logger.info(f'Best CatBoost Classifier Model: {catboost_tuner}')

        cb_iterations = int(best_catboost['iterations'])
        cb_learning_rate = best_catboost['learning_rate']
        cb_depth = int(best_catboost['depth'])
        # Evaluate the CatBoost model
        roc_auc_cb, f1_score_cb, runtime_cb = model_cb(X, y, cb_iterations, cb_learning_rate, cb_depth, scorer, kf)
        append_metrics('CATBoost', cb_iterations, roc_auc_cb, f1_score_cb, runtime_cb)
        _logger.info(
            f'CatBoost Evaluation: ROC AUC Score= {roc_auc_cb}, F1 Score= {f1_score_cb}, Runtime= {runtime_cb}')
    except Exception as e:
        _logger.error(f'Error evaluating CatBoost model:{e}')

    '''
    ================================
    UNSUPERVISED LEARNING ALGORITHMS
    ================================
    This section evaluates the performance of various unsupervised learning algorithms, incl. LOF(Local Outlier Factor),
    Isolation Forest, PCA(Principal Component Analysis), DBSCAN, COPOD(Copula-Based Outlier Detection), and 
    ECOD(Empirical Cumulative Distribution Based Outlier Detection). Some of the algorithms have been fine-tuned 
    using the hyperopt library, while the others use default parameters provided by sklearn library.
    '''
    try:
        # MODEL LOCAL OUTLIER FACTOR (LOF)
        _logger.info('Starting LOF Evaluation')
        # Tune the LOF model to get the best hyperparameters
        lof_tune = LOF_tuner(X_scaled, y)
        lof_estimator = lof_tune.tune_model()
        _logger.info(f'Best n_neighbors for LOF Model: {lof_estimator}')
        # Evaluate the LOF model
        roc_auc_lof, f1_score_lof, precision_lof, recall_lof, accuracy_lof, runtime_lof = model_lof(X_scaled, y, lof_estimator)
        append_metrics('LOF', lof_estimator, roc_auc_lof, f1_score_lof, runtime_lof)
        unsupervised_metrics('LOF', lof_estimator, roc_auc_lof, f1_score_lof, precision_lof, recall_lof, accuracy_lof, runtime_lof)
        _logger.info(
            f'LOF Evaluation: ROC AUC Score={roc_auc_lof}, F1 Score={f1_score_lof}, Precision Score={precision_lof}, \n'
            f'Recall Score={recall_lof}, Accuracy Score={accuracy_lof}, Runtime={runtime_lof}')
    except Exception as e:
        _logger.error(f'Error evaluating LOF model:{e}')

    try:
        # MODEL PRINCIPAL COMPONENT ANALYSIS (PCA)
        _logger.info('Starting PCA Evaluation')
        # Evaluate the PCA model
        roc_auc_pca, f1_score_pca, precision_pca, recall_pca, accuracy_pca, runtime_pca = model_pca(X_scaled, y)
        append_metrics('PCA', None, roc_auc_pca, f1_score_pca, runtime_pca)
        unsupervised_metrics('PCA', None, roc_auc_pca, f1_score_pca, precision_pca, recall_pca, accuracy_pca, runtime_pca)
        _logger.info(
            f'PCA Evaluation: ROC AUC Score={roc_auc_pca}, F1 Score={f1_score_pca}, Precision Score={precision_pca},\n'
            f'Recall Score={recall_pca}, Accuracy Score={accuracy_pca}, Runtime={runtime_pca}')
    except Exception as e:
        _logger.error(f'Error evaluating PCA model:{e}')

    try:
        # MODEL ISOLATION FOREST (IF)
        _logger.info('Starting Isolation Forest Evaluation')
        # Tune the Isolation Forest model to get the best hyperparameters
        forest_tunner = IsolationForest_tuner(X, y)
        forest_estimator = forest_tunner.tune_model()
        _logger.info(f'Best Isolation Forest Model: {forest_estimator}')
        # Evaluate the IF model
        roc_auc_if, f1_score_if, precision_if, recall_if, accuracy_if, runtime_if = model_iforest(X, y, forest_estimator)
        append_metrics('Isolation Forest', forest_estimator, roc_auc_if, f1_score_if, runtime_if)
        unsupervised_metrics('Isolation Forest', forest_estimator, roc_auc_if, f1_score_if, precision_if, recall_if, accuracy_if, runtime_if)
        _logger.info(
            f'Isolation Forest Evaluation: ROC AUC Score={roc_auc_if}, F1 Score={f1_score_if}, Precision Score={precision_if}, \n'
            f'Recall Score={recall_if}, Accuracy Score={accuracy_if}, Runtime={runtime_if}')
    except Exception as e:
        _logger.error(f'Error evaluating Isolation Forest model:{e}')

    try:
        # MODEL DBSCAN
        _logger.info('Starting DBSCAN Evaluation')
        # Tune the DBSCAN model to get the best hyperparameters
        # Code for hyper tune DBSCAN
        dbscan_tuner = DBSCAN_tuner(X_scaled, y)
        dbscan_cluster = dbscan_tuner.tune_model()
        _logger.info(f'Best DBSCAN Model: {dbscan_cluster}')
        # Save the best parameters
        best_eps = dbscan_cluster['eps']
        best_min_samples = int(dbscan_cluster['min_samples'])
        # Evaluate the DBSCAN model
        roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan = model_dbscan(
            X_scaled, y, eps=best_eps, min_samples=best_min_samples)
        append_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, runtime_dbscan)
        unsupervised_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan)
        _logger.info(
            f'DBSCAN Evaluation: ROC AUC Score={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, Precision Score={precision_dbscan}, \n'
            f'Recall Score={recall_dbscan}, Accuracy Score={accuracy_dbscan}, Runtime={runtime_dbscan}')
    except Exception as e:
        _logger.error(f'Error evaluating DBSCAN model:{e}')

    try:
        # MODEL COPULA BASED OUTLIER DETECTION (COPOD)
        _logger.info('Starting COPOD Evaluation')
        # Evaluate the COPOD model
        roc_auc_copod, f1_score_copod, precision_copod, recall_copod, accuracy_copod, runtime_copod = model_copod(X, y)
        append_metrics('COPOD', None, roc_auc_copod, f1_score_copod, runtime_copod)
        unsupervised_metrics('COPOD', None, roc_auc_copod, f1_score_copod, precision_copod, recall_copod,
                        accuracy_copod, runtime_copod)
        _logger.info(
            f'COPOD Evaluation: ROC AUC Score={roc_auc_copod}, F1 Score={f1_score_copod}, Precision Score={precision_copod} \n'
            f'Recall Score={recall_copod}, Accuracy Score={accuracy_copod}, Runtime={runtime_copod}')
    except Exception as e:
        _logger.error(f'Error evaluating COPOD model:{e}')

    try:
        # MODEL EMPIRICAL CUMULATIVE DISTRIBUTION BASED OUTLIER DETECTION (ECOD)
        _logger.info('Starting ECOD Evaluation')
        # Evaluate the ECOD model
        roc_auc_ecod, f1_score_ecod, precision_ecod, recall_ecod, accuracy_ecod, runtime_ecod = model_ecod(X, y)
        append_metrics('ECOD', None, roc_auc_ecod, f1_score_ecod, runtime_ecod)
        unsupervised_metrics('ECOD', None, roc_auc_ecod, f1_score_ecod, precision_ecod, recall_ecod, accuracy_ecod,
                            runtime_ecod)
        _logger.info(
            f'ECOD Evaluation: ROC AUC Score={roc_auc_ecod}, F1 Score={f1_score_ecod}, Precision Score={precision_ecod}\n'
            f'Recall Score={recall_ecod}, Accuracy Score={accuracy_ecod}, Runtime={runtime_ecod}')
    except Exception as e:
        _logger.error(f'Error evaluating ECOD model:{e}')

    # In this section, the results of the evaluation are saved and used to create the necessary visualizations.

    # Create a dataframe to store the evaluation metrics
    metrics_df = pd.DataFrame(metrics)
    unsupervised_metrics_df = pd.DataFrame(metrics_unsupervised)

    # Save the unsupervised metrics to a CSV file
    metrics_df.to_csv('../results/performance/Metrics(DS3).csv', index=False)
    _logger.info('The evaluation results are saved to CSV file.')

    unsupervised_metrics_df.to_csv('../results/performance/Unsupervised_Metrics(DS3).csv', index=False)
    _logger.info('The evaluation results for unsupervised learning are saved to CSV file.')

    # Visualizing the results ROC-AUC Score - Runtime
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['Runtime'], metrics_df['ROC_AUC_Score'], color='blue', s=100)
    texts = []
    for i, txt in enumerate(metrics_df['Model']):
        texts.append(plt.text(metrics_df['Runtime'][i], metrics_df['ROC_AUC_Score'][i], txt, fontsize=12))
    adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'))
    plt.grid(True)
    plt.xlabel('Runtime', fontsize=14, fontweight='bold')
    plt.ylabel('ROC AUC Score', fontsize=14, fontweight='bold')
    plt.title('ROC AUC Score vs Runtime comparison', fontsize=16, fontweight='bold')
    plt.savefig('../results/performance/ROC_AUC_vs_Runtime(DS3).png', bbox_inches='tight')
    plt.show()

    # Visualizing the results F1 Score - Runtime
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['Runtime'], metrics_df['F1_Score'], color='blue', s=100)
    texts = []
    for i, txt in enumerate(metrics_df['Model']):
        texts.append(plt.text(metrics_df['Runtime'][i], metrics_df['F1_Score'][i], txt, fontsize=12))
    adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'))
    plt.grid(True)
    plt.xlabel('Runtime', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.title('F1 Score vs Runtime comparison', fontsize=16, fontweight='bold')
    plt.savefig('../results/performance/F1_Score_vs_Runtime(DS3).png', bbox_inches='tight')
    plt.show()

