import pandas as pd
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import time
from pyod.utils.data import evaluate_print
from models.supervised_learning import model_knn
from models.supervised_learning import model_xgboost
from models.supervised_learning import model_svm

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

'''
model_name = 'KNN'
model = KNN( n_neighbors=5, contamination=0.1)
model.fit(X_train)

# get the prediction labels and outlier scores of the training data
y_train_pred = model.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = model.decision_scores_  # raw outlier scores

# get the prediction on the test data
y_test_pred = model.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = model.decision_function(X_test)  # outlier scores

roc_auc = roc_auc_score(y_test, y_test_pred)
print('ROC AUC Score: {}'.format(roc_auc))
f1_score = f1_score(y_test, y_test_pred, average='weighted')
print('F1 Score: {}'.format(f1_score))'''

#results_knn = model_knn(X_train, X_test, y_train, y_test, 5)
#results_xgboost = model_xgboost(X_train, X_test, y_train, y_test, 4)
results_svm = model_svm(X_train, X_test, y_train, y_test)