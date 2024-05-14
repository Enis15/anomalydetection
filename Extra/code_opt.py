import pandas as pd
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

# Supervised learning models
from utils.supervised_learning import model_knn, model_xgboost, model_svm, model_cb, model_nb, model_rf
# Unsupervised learning models
from utils.unsupervised_learning import model_lof, model_iforest, model_ecod, model_pca, model_kmeans, model_copod
# Hyperparameter tuning functions
from utils.paramet_tune import paramet_tune, Catboost_tune, LOF_tune, Kmeans_tune

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

# Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if __name__ == '__main__':
    # Ensure compatibility for multiprocessing
    freeze_support()

    # DataFrame to store metrics
    metrics = []

    # Function to append metrics to the list
    def append_metrics(model_name, estimator, roc_auc, f1_score, runtime):
        metrics.append({
            'Model': model_name,
            'Estimator': estimator,
            'ROC_AUC': roc_auc,
            'F1_Score': f1_score,
            'Runtime': runtime
        })

    '''
    ===============================
    SUPERVISED LEARNING ALGORITHMS
    ===============================
    '''

    # K-Nearest Neighbors (KNN)
    best_knn_model = paramet_tune(X_train, y_train, model_name='knn')
    k_value = best_knn_model['learner'].n_neighbors
    roc_auc_knn, f1_score_knn, runtime_knn = model_knn(X_train, X_test, y_train, y_test, k_value)
    append_metrics('KNN', k_value, roc_auc_knn, f1_score_knn, runtime_knn)

    # Random Forest (RF)
    best_rf_model = paramet_tune(X_train, y_train, model_name='random_forest')
    rf_value = best_rf_model['learner'].n_estimators
    roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_value)
    append_metrics('Random Forest', rf_value, roc_auc_rf, f1_score_rf, runtime_rf)

    # XGBoost
    best_xgboost_model = paramet_tune(X_train, y_train, model_name='xgboost')
    xgboost_value = best_xgboost_model['learner'].n_estimators
    xgboost_depth = best_xgboost_model['learner'].max_depth
    roc_auc_xgboost, f1_score_xgboost, runtime_xgboost = model_xgboost(X_train, X_test, y_train, y_test, xgboost_value, xgboost_depth)
    append_metrics('XGBoost', (xgboost_value, xgboost_depth), roc_auc_xgboost, f1_score_xgboost, runtime_xgboost)

    # Support Vector Machine (SVM)
    roc_auc_svm, f1_score_svm, runtime_svm = model_svm(X_train, X_test, y_train, y_test)
    append_metrics('SVM', None, roc_auc_svm, f1_score_svm, runtime_svm)

    # Naive Bayes (NB)
    roc_auc_nb, f1_score_nb, runtime_nb = model_nb(X_train, X_test, y_train, y_test)
    append_metrics('Naive Bayes', None, roc_auc_nb, f1_score_nb, runtime_nb)

    # CatBoost (CB)
    catboost_tuner = Catboost_tune(X_train, X_test, y_train, y_test)
    best_catboost = catboost_tuner.tune_model()
    cb_iterations = int(best_catboost['iterations'])
    cb_learning_rate = best_catboost['learningrate']
    cb_depth = int(best_catboost['depth'])
    roc_auc_cb, f1_score_cb, runtime_cb = model_cb(X_train, X_test, y_train, y_test, cb_iterations, cb_learning_rate, cb_depth)
    append_metrics('CatBoost', (cb_iterations, cb_learning_rate, cb_depth), roc_auc_cb, f1_score_cb, runtime_cb)

    '''
    ================================
    UNSUPERVISED LEARNING ALGORITHMS
    ================================
    '''

    # Local Outlier Factor (LOF)
    lof_tune = LOF_tune(X, y)
    k_lof = lof_tune.tune_model()
    roc_auc_lof, f1_score_lof, runtime_lof = model_lof(X, y, k_lof)
    append_metrics('LOF', k_lof, roc_auc_lof, f1_score_lof, runtime_lof)

    # Principal Component Analysis (PCA)
    roc_auc_pca, f1_score_pca, runtime_pca = model_pca(X, y)
    append_metrics('PCA', None, roc_auc_pca, f1_score_pca, runtime_pca)

    # Isolation Forest (IF)
    best_if_model = paramet_tune(X_train, y_train, model_name='isolation_forest')
    if_value = best_if_model['learner'].n_estimators
    roc_auc_if, f1_score_if, runtime_if = model_iforest(X, y, if_value)
    append_metrics('IForest', if_value, roc_auc_if, f1_score_if, runtime_if)

    # K-Means
    k_means_tuner = Kmeans_tune(X, y)
    k_clusters = k_means_tuner.tune_model()
    roc_auc_kmeans, f1_score_kmeans, runtime_kmeans = model_kmeans(X, y, k_clusters)
    append_metrics('K-Means', k_clusters, roc_auc_kmeans, f1_score_kmeans, runtime_kmeans)

    # Copula-Based Outlier Detection (COPOD)
    roc_auc_copod, f1_score_copod, runtime_copod = model_copod(X, y)
    append_metrics('COPOD', None, roc_auc_copod, f1_score_copod, runtime_copod)

    # Empirical Cumulative Distribution Based Outlier Detection (ECOD)
    roc_auc_ecod, f1_score_ecod, runtime_ecod = model_ecod(X, y)
    append_metrics('ECOD', None, roc_auc_ecod, f1_score_ecod, runtime_ecod)

    # Create a DataFrame to store the evaluation metrics
    metrics_df = pd.DataFrame(metrics)

    # Save the metrics to a CSV file
    metrics_df.to_csv('pythonProject/results/Metrics(DS1).csv', index=False)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['Runtime'], metrics_df['ROC_AUC'], color='blue')
    for i, txt in enumerate(metrics_df['Model']):
        plt.annotate(txt, (metrics_df['Runtime'][i], metrics_df['ROC_AUC'][i]))
    plt.xlabel('Runtime')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs Runtime comparison')
    plt.show()
    plt.savefig('./ROC_AUC_vs_Runtime.png', bbox_inches='tight')
