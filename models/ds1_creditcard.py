from utils.supervised_learning import model_knn, model_xgboost, model_svm, model_cb, model_nb, model_rf
from utils.unsupervised_learning import model_lof, model_iforest, model_ecod, model_pca, model_kmeans, model_copod
from utils.paramet_tune import paramet_tune, Catboost_tune, LOF_tune
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from multiprocessing import freeze_support
import matplotlib.pyplot as plt

'''
Dataset description
'''
#Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

#Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if __name__ == '__main__':
    #Ensure compatability
    freeze_support()

    '''
    ===============================
    SUPERVISED LEARNING ALGORITHMS
    ===============================
    
    This section of the code evaluates the performance of supervised learning algorithms, incl. KNN(K-Nearest Neighbor), Random Forest,
    XGBoost(Extreme Gradient Boosting), SVM(Support Vector Machine), Naive Bayes and CATBoost(Categorical Boosting). 
    Some of the algorithms have been fine-tuned using the hyperopt/hpsklearn library, while the others use default parameters
    provided by sklearn library.
    '''
    #MODEL K-NEAREST NEIGHBORS (KNN)
    #Tune the KNN model to get the best hyperparameters
    best_knn_model = paramet_tune(X_train, y_train, model_name='knn')
    print(best_knn_model) #Get the results of parameter tuning
    k_value = best_knn_model['learner'].n_neighbors  #Save the value of k

    #Evaluate the KNN model using the best parameters
    roc_auc_knn, f1_score_knn, runtime_knn = model_knn(X_train, X_test, y_train, y_test, k_value)

    #MODEL RANDOM FOREST (RF)
    # Tune the Random Forest model to get the best hyperparameters
    best_rf_model = paramet_tune(X_train, y_train, model_name='random_forest')
    print(best_rf_model)  # Get the results of parameter tuning
    rf_value = best_rf_model['learner'].n_estimators  # Save the value of n_estimators

    #Evaluate the KNN model using the best parameters
    roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_value) 

    #MODEL XGBOOST
    # Tune the XGBOOST model to get the best hyperparameters
    best_xgboost_model = paramet_tune(X_train, y_train, model_name='xgboost')
    print(best_xgboost_model)  # Get the results of parameter tuning

    xgboost_value = best_xgboost_model['learner'].n_estimators #Save the value of n_estimators
    xgboost_depth = best_xgboost_model['learner'].max_depth #Save the value of max_depth

    #Evaluate the XGBoost model using the best parameters
    roc_auc_xgboost, f1_score_xgboost, runtime_xgboost = model_xgboost(X_train, X_test, y_train, y_test, xgboost_value, xgboost_depth)

    #MODEL SUPPORT VECTOR MACHINE (SVM)
    #Evaluate the SVM model
    roc_auc_svm, f1_score_svm, runtime_svm = model_svm(X_train, X_test, y_train, y_test)

    #MODEL NAIVE BAYES (NB)
    #Evaluate the Naive Bayes model
    roc_auc_nb, f1_score_nb, runtime_nb = model_nb(X_train, X_test, y_train, y_test)

    #MODEL CATBOOST (CB)
    #Tune the CatBoost model to get the best hyperparameters
    catboost_tuner = Catboost_tune(X_train, X_test, y_train, y_test)
    best_catboost = catboost_tuner.tune_model()

    cb_iterations = int(best_catboost['iterations'])
    cb_learning_rate = best_catboost['learningrate']
    cb_depth = int(best_catboost['depth'])
    #Evaluate the CatBoost model
    roc_auc_cb, f1_score_cb, runtime_cb = model_cb(X_train, X_test, y_train, y_test, cb_iterations, cb_learning_rate, cb_depth)

    '''
    ================================
    UNSUPERVISED LEARNING ALGORITHMS
    ================================
    This section evaluates the performance of various unsupervised learning algorithms, incl. LOF(Local Outlier Factor), Isolation Forest,
    PCA(Principal Component Analysis), K-Means, COPOD(Copula-Based Outlier Detection), and ECOD(Empirical Cumulative Distribution Based Outlier Detection). 
    Some of the algorithms have been fine-tuned using the hyperopt/hpsklearn library,
    while the others use default parameters provided by sklearn library.
    '''

    #MODEL LOCAL OUTLIER FACTOR (LOF)
    #Tune the LOF model to get the best hyperparameters
    lof_tune = LOF_tune(X, y)
    k_lof = lof_tune.tune_model()

    #Evaluate the LOF model
    roc_auc_lof, f1_score_lof, runtime_lof = model_lof(X, y, k_lof)

    #MODEL PRINCIPAL COMPONENT ANALYSIS (PCA)

    #Evaluate the PCA model
    roc_auc_pca, f1_score_pca, runtime_pca = model_pca(X, y)

    #MODEL ISOLATION FOREST (IF)
    # Tune the Isolation Forest model to get the best hyperparameters
    best_if_model = paramet_tune(X_train, y_train, model_name='isolation_forest')
    print(best_if_model)  # Get the results of parameter tuning
    if_value = best_if_model['learner'].n_estimators  # Save the value of n_estimators

    #Evaluate the IF model
    roc_auc_if, f1_score_if, runtime_if = model_iforest(X, y, if_value)

    #MODEL CLUSTER BASED LOCAL OUTLIER FACTOR (K-Means)
    #Tune the K-Means model to get the best hyperparameters
    #Code for hyper tune K-Means
    k_means = 8
    #Evaluate the K-Means model
    roc_auc_kmeans, f1_score_kmeans, runtime_kmeans = model_kmeans(X, y, k_means)

    #MODEL COPULA BASED OUTLIER DETECTION (COPOD)

    #Evaluate the COPOD model
    roc_auc_copod, f1_score_copod, runtime_copod = model_copod(X, y)

    #MODEL EMPIRICAL CUMULATIVE DISTRIBUTION BASED OUTLIER DETECTION (ECOD)

    #Evaluate the ECOD model
    roc_auc_ecod, f1_score_ecod, runtime_ecod = model_ecod(X, y)

    #In this section, the results of the evaluation are saved and used to create the necessary visualizations.

    #Create a dataframe to store the evaluation metrics
    metrics = pd.DataFrame({
        'Model': ['KNN', 'Random Forest', 'XGBoost', 'SVM', 'Naive Bayes', 'CatBoost', 'LOF', 'PCA', 'IForest', 'K-Means', 'COPOD', 'ECOD'],
        'Estimator': [k_value, rf_value, xgboost_value, '', '', cb_iterations , k_lof, '', if_value, k_means, '', ''],
        'ROC_AUC': [roc_auc_knn, roc_auc_rf, roc_auc_xgboost, roc_auc_svm, roc_auc_nb, roc_auc_cb. roc_auc_lof, roc_auc_pca, roc_auc_if, roc_auc_kmeans, roc_auc_copod, roc_auc_ecod],
        'F1_Score': [f1_score_knn, f1_score_rf, f1_score_xgboost, f1_score_svm, f1_score_nb, f1_score_cb, f1_score_lof, f1_score_pca, f1_score_if, f1_score_kmeans, f1_score_copod, f1_score_ecod],
        'Runtime': [runtime_rf, runtime_rf, runtime_xgboost, runtime_svm, runtime_nb, runtime_cb, runtime_lof, runtime_pca, runtime_if, runtime_kmeans,runtime_copod, runtime_ecod]
    })

    #Save the metrics to a CSV file
    metrics.to_csv('pythonProject/results/Metrics(DS1).csv', index=False)

    #Visualizing the results
    plt.figure(figsize=(10, 6))
    plt.scatter(metrics['Runtime'], metrics['ROC_AUC'], color='blue')
    for i, txt in enumerate(metrics['Model']):
        plt.annotate(txt, (metrics['Runtime'][i], metrics['ROC_AUC'][i]))
    plt.xlabel('Runtime')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs Runtime comparison')
    plt.show()
    #plt.savefig('./ROC_AUC_vs_Runtime.png', bbox_inches='tight')
