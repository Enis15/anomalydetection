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
 # Hyperparameter tuning functions
from utils.paramet_tune import IsolationForest_tuner, KNN_tuner, XGBoost_tuner, Catboost_tuner, LOF_tuner, \
    RandomForest_tuner, DBSCAN_tuner

# Initialize the logger
_logger = logger(__name__)

'''
Dataset 1: Credit card transaction, with over 1.200.0000 records and 24 features.
'''
# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

# Drop irrelavant features
df = df.drop(['Unnamed: 0', 'trans_date_trans_time', 'trans_num', 'unix_time', 'dob', 'first', 'last', 'merch_zipcode'],
             axis=1)

# Encoding categorical features with numerical variables
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

# Determining the X and y values
X = df.drop('is_fraud', axis=1)
y = df['is_fraud'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize the data

# Setting the fold splits for unsupervised learning models
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
        roc_auc_xgboost, f1_score_xgboost, runtime_xgboost = model_xgboost(X, y, xgboost_value, xgboost_depth,
                                                                           xgboost_learn_rate, scorer, kf)
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