'''
from data import data_preprocessing
from pyod.utils.pca import PCA
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
import pandas as pd

def anomaly_detection(df, model_name):
    # Load data and initialize the model
    clf = PCA()
    X_train, X_test, y_train, y_test = data_preprocessing(df, 'Transaction Amount')

    # Measure execution time
    start_time = time.time()

    # Fit the model
    clf.fit(X_train)

    # Get model scores
    y_train_scores = clf.decision_scores_

    # Runtime evaluation
    execution_time = time.time() - start_time
    print('Run time is:', round(execution_time, 3), 'seconds')

    # Evaluate the model
    roc_auc = roc_auc_score(y_test, clf.predict(X_test))

# Load data
df = pd.read_csv('../Datasets_manipulation/data/Unlabeled_DS/cust_trans.csv')

# Define model name
clf_name = 'PCA'

# Perform anomaly detection
anomaly_detection(df, clf_name) '''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pyod.models.pca import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# Function for data preprocessing
def data_preprocessing(df, target_column, test_size=0.3, random_state=None):
    # Drop missing values
    df.dropna(inplace=True)

    # Split the target variable
    X = df[target_column]
    y = df.drop(columns=[target_column])

    # Encode categorical variables
    X[X.select_dtypes(include='object').columns] = X[X.select_dtypes(include='object').columns].apply(LabelEncoder().fit_transform)

    # Scale numerical features
    scaler = StandardScaler()
    X[X.select_dtypes(include=['float', 'int']).columns] = scaler.fit_transform(X.select_dtypes(include=['float', 'int']))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

# Function for anomaly detection
def anomaly_detection(df, target_column, threshold):
    # Load data and initialize the model
    clf = PCA()
    X_train, X_test, y_train, y_test = data_preprocessing(df, target_column)

    # Measure execution time
    start_time = time.time()

    # Fit the model
    clf.fit(X_train)

    # Get anomaly scores
    anomaly_scores = clf.decision_function(X_test)

    # Runtime evaluation
    execution_time = time.time() - start_time
    print('Run time is:', round(execution_time, 3), 'seconds')

    # Define anomaly threshold
    threshold = threshold

    # Classify anomalies based on the threshold
    y_pred = (y_test > threshold).astype(int)

    # Evaluate the model using ROC AUC score
    roc_auc = roc_auc_score(y_pred, anomaly_scores)
    recall = recall_score(y_pred, anomaly_scores)
    accuracy = accuracy_score(y_pred, anomaly_scores)
    precision = precision_score(y_pred, anomaly_scores)
    f1 = f1_score(y_pred, anomaly_scores)


    print('ROC AUC Score:', roc_auc)
    print('Accuracy Score:', accuracy)
    print('Precision Score:', precision)
    print('Recall Score:', recall)
    print('F1 Score:', f1)

# Load data
df = pd.read_csv('../../data/datasets/Unlabeled_DS/cust_trans.csv')

# Perform anomaly detection
anomaly_detection(df, 'Transaction Amount', 1000)

