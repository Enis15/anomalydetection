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
Dataset 4: Metaverse Financial Transaction Dataset, with over 78.000 records and 14 features.
'''

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv')

# Dropping irrelevant columns for the anomaly detection
df = df.drop(['timestamp', 'sending_address', 'receiving_address'], axis=1)

# Relabeling column target column 'anomaly', where low risk:0, moderate & high risk =1
pd.set_option('future.no_silent_downcasting', True) # Ensure downcasting behavior is consistent with future versions of pandas
df['anomaly'] = df['anomaly'].replace({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})
df['anomaly'] = df['anomaly'].astype(int)

# Encoding categorical features with numerical variables
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

# Determining the X and y values
X = df.drop('anomaly', axis=1)
y = df['anomaly'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardize the data

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
    #append_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, runtime_dbscan)
    #unsupervised_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan)
    _logger.info(
        f'DBSCAN Evaluation: ROC AUC Score={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, Precision Score={precision_dbscan}, \n'
        f'Recall Score={recall_dbscan}, Accuracy Score={accuracy_dbscan}, Runtime={runtime_dbscan}')
except Exception as e:
    _logger.error(f'Error evaluating DBSCAN model:{e}')


