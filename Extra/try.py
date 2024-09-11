import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
from utils.paramet_tune import paramet_tune, Catboost_tuner, LOF_tuner, DBSCAN_tuner, RandomForest_tuner

# Initialize the logger
_logger = logger(__name__)

'''
Dataset description
'''

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv')

# Dropping irrelevant columns for the anomaly detection
df = df.drop(['timestamp', 'sending_address', 'receiving_address'], axis=1)

# Encoding categorical features
columns_label = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
for i in columns_label:
    label = LabelEncoder()
    df[i] = label.fit_transform(df[i])

# Relabeling column target column 'anomaly', where low risk:0, moderate & high risk =1
pd.set_option('future.no_silent_downcasting', True) # Ensure downcasting behavior is consistent with future versions of pandas
df['anomaly'] = df['anomaly'].replace({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})
df['anomaly'] = df['anomaly'].astype(int)

# Determining the X and y values
X = df.drop('anomaly', axis=1)
y = df['anomaly'].values

# Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

metrics = []

# Function to append results to metrics list
def append_metrics(modelname, estimator, roc_auc, f1_score, runtime):
    metrics.append({
        'Model': modelname,
        'Estimator': estimator,
        'ROC_AUC_Score': roc_auc,
        'F1_Score': f1_score,
        'Runtime': runtime
    })

try:
    # MODEL DBSCAN
    _logger.info('Starting DBSCAN Evaluation')
    # Tune the DBSCAN model to get the best hyperparameters
    # Code for hyper tune DBSCAN
    dbscan_tuner = DBSCAN_tuner(X, y)
    dbscan_cluster = dbscan_tuner.tune_model()
    _logger.info(f'Best K-Means Model: {dbscan_cluster}')

    best_eps = dbscan_cluster['eps']
    best_min_samples = int(dbscan_cluster['min_samples'])
    # Evaluate the DBSCAN model
    roc_auc_dbscan, f1_score_dbscan, runtime_dbscan = model_dbscan(X, y, eps=best_eps, min_samples=best_min_samples)
    append_metrics('DBSCAN', best_eps, roc_auc_dbscan, f1_score_dbscan, runtime_dbscan)
    _logger.info(
        f'DBSCAN Evaluation: ROC AUC Score={roc_auc_dbscan}, F1 Score={f1_score_dbscan}, Runtime={runtime_dbscan}')
except Exception as e:
    _logger.error(f'Error evaluating DBSCAN model:{e}')