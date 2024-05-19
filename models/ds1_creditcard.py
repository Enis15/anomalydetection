import pandas as pd
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

# Supervised learning models
from utils.supervised_learning import model_knn, model_xgboost, model_svm, model_cb, model_nb, model_rf
# Unsupervised learning models
from utils.unsupervised_learning import model_lof, model_iforest, model_ecod, model_pca, model_kmeans, model_copod
# Hyperparameter tuning functions
from utils.paramet_tune import paramet_tune, Catboost_tuner, LOF_tuner, Kmeans_tuner, RandomForest_tuner


'''
Dataset description
'''
# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

# Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

# Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if __name__ == '__main__':
    # Ensure compatability
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

    '''
    ===============================
    SUPERVISED LEARNING ALGORITHMS
    ===============================
    
    This section of the code evaluates the performance of supervised learning algorithms, incl. KNN(K-Nearest Neighbor), Random Forest Classifier,
    XGBoost(Extreme Gradient Boosting), SVM(Support Vector Machine), Naive Bayes and CATBoost(Categorical Boosting). 
    Some of the algorithms have been fine-tuned using the hyperopt/hpsklearn library, while the others use default parameters
    provided by sklearn library.
    '''
    # MODEL K-NEAREST NEIGHBORS (KNN)
    # Tune the KNN model to get the best hyperparameters
    best_knn_model = paramet_tune(X_train, y_train, model_name='knn')
    print(best_knn_model) #Get the results of parameter tuning
    k_value = best_knn_model['learner'].n_neighbors  #Save the value of k
    # Evaluate the KNN model using the best parameters
    roc_auc_knn, f1_score_knn, runtime_knn = model_knn(X_train, X_test, y_train, y_test, k_value)
    append_metrics('KNN', k_value, roc_auc_knn, f1_score_knn, runtime_knn)

    # MODEL RANDOM FOREST (RF)
    # Tune the Random Forest model to get the best hyperparameters
    rf_tuner = RandomForest_tuner(X_train, X_test, y_train, y_test)
    best_rf_model = rf_tuner.tune_model()

    rf_estimator = int(best_rf_model['n_estimators'])
    rf_depth = int(best_rf_model['max_depth'])

    # Evaluate the Random Forest Classifier using the best parameters
    roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_estimator, rf_depth)
    append_metrics('Random Forest Classifier', rf_estimator, roc_auc_rf, f1_score_rf, runtime_rf)

    # MODEL XGBOOST
    # Tune the XGBOOST model to get the best hyperparameters
    best_xgboost_model = paramet_tune(X_train, y_train, model_name='xgboost')
    print(best_xgboost_model)  # Get the results of parameter tuning

    xgboost_value = best_xgboost_model['learner'].n_estimators #Save the value of n_estimators
    xgboost_depth = best_xgboost_model['learner'].max_depth #Save the value of max_depth
    xgboost_learningrate = best_xgboost_model['learner'].learning_rate

    # Evaluate the XGBoost model using the best parameters
    roc_auc_xgboost, f1_score_xgboost, runtime_xgboost = model_xgboost(X_train, X_test, y_train, y_test, xgboost_value, xgboost_depth, xgboost_learningrate)
    append_metrics('XGBoost', xgboost_value, roc_auc_xgboost, f1_score_xgboost, runtime_xgboost)

    # MODEL SUPPORT VECTOR MACHINE (SVM)
    # Evaluate the SVM model
    roc_auc_svm, f1_score_svm, runtime_svm = model_svm(X_train, X_test, y_train, y_test)
    append_metrics('SVM', None, roc_auc_svm, f1_score_svm, runtime_svm)

    # MODEL NAIVE BAYES (NB)
    # Evaluate the Naive Bayes model
    roc_auc_nb, f1_score_nb, runtime_nb = model_nb(X_train, X_test, y_train, y_test)

    # MODEL CATBOOST (CB)
    # Tune the CatBoost model to get the best hyperparameters
    catboost_tuner = Catboost_tuner(X_train, X_test, y_train, y_test)
    best_catboost = catboost_tuner.tune_model()

    cb_iterations = int(best_catboost['iterations'])
    cb_learning_rate = best_catboost['learning_rate']
    cb_depth = int(best_catboost['depth'])
    # Evaluate the CatBoost model
    roc_auc_cb, f1_score_cb, runtime_cb = model_cb(X_train, X_test, y_train, y_test, cb_iterations, cb_learning_rate, cb_depth)
    append_metrics('CATBoost', cb_iterations, roc_auc_cb, f1_score_cb, runtime_cb)

    '''
    ================================
    UNSUPERVISED LEARNING ALGORITHMS
    ================================
    This section evaluates the performance of various unsupervised learning algorithms, incl. LOF(Local Outlier Factor), Isolation Forest,
    PCA(Principal Component Analysis), K-Means, COPOD(Copula-Based Outlier Detection), and ECOD(Empirical Cumulative Distribution Based Outlier Detection). 
    Some of the algorithms have been fine-tuned using the hyperopt/hpsklearn library,
    while the others use default parameters provided by sklearn library.
    '''

    # MODEL LOCAL OUTLIER FACTOR (LOF)
    # Tune the LOF model to get the best hyperparameters
    lof_tune = LOF_tuner(X, y)
    k_lof = lof_tune.tune_model()

    # Evaluate the LOF model
    roc_auc_lof, f1_score_lof, runtime_lof = model_lof(X, y, k_lof)
    append_metrics('LOF', k_lof, roc_auc_lof, f1_score_lof, runtime_lof)

    # MODEL PRINCIPAL COMPONENT ANALYSIS (PCA)
    # Evaluate the PCA model
    roc_auc_pca, f1_score_pca, runtime_pca = model_pca(X, y)
    append_metrics('PCA', None, roc_auc_pca, f1_score_pca, runtime_pca)

    # MODEL ISOLATION FOREST (IF)
    # Tune the Isolation Forest model to get the best hyperparameters
    best_if_model = paramet_tune(X_train, y_train, model_name='isolation_forest')
    print(best_if_model)  # Get the results of parameter tuning
    if_value = best_if_model['learner'].n_estimators  # Save the value of n_estimators

    # Evaluate the IF model
    roc_auc_if, f1_score_if, runtime_if = model_iforest(X, y, if_value)
    append_metrics('Isolation Forest', if_value, roc_auc_if, f1_score_if, runtime_if)

    # MODEL CLUSTER BASED LOCAL OUTLIER FACTOR (K-Means)
    # Tune the K-Means model to get the best hyperparameters
    # Code for hyper tune K-Means
    k_means_tuner = Kmeans_tuner(X, y)
    k_clusters = k_means_tuner.tune_model()
    # Evaluate the K-Means model
    roc_auc_kmeans, f1_score_kmeans, runtime_kmeans = model_kmeans(X, y, k_clusters)
    append_metrics('K-Means', k_clusters, roc_auc_kmeans, f1_score_kmeans, runtime_kmeans)

    # MODEL COPULA BASED OUTLIER DETECTION (COPOD)
    # Evaluate the COPOD model
    roc_auc_copod, f1_score_copod, runtime_copod = model_copod(X, y)
    append_metrics('COPOD', None, roc_auc_copod, f1_score_copod, runtime_copod)

    # MODEL EMPIRICAL CUMULATIVE DISTRIBUTION BASED OUTLIER DETECTION (ECOD)
    # Evaluate the ECOD model
    roc_auc_ecod, f1_score_ecod, runtime_ecod = model_ecod(X, y)
    append_metrics('ECOD', None, roc_auc_ecod, f1_score_ecod, runtime_ecod)

    # In this section, the results of the evaluation are saved and used to create the necessary visualizations.

    # Create a dataframe to store the evaluation metrics
    metrics_df = pd.DataFrame(metrics)

    # Save the metrics to a CSV file
    metrics_df.to_csv('./Metrics(DS1).csv', index=False)

    # Visualizing the results
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics_df['Runtime'], metrics_df['ROC_AUC_Score'], color='blue')
    for i, txt in enumerate(metrics_df['Model']):
        plt.annotate(txt, (metrics_df['Runtime'][i], metrics_df['ROC_AUC_Score'][i]))
    plt.xlabel('Runtime')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs Runtime comparison')
    plt.show()
    plt.savefig('./ROC_AUC_vs_Runtime.png', bbox_inches='tight')
