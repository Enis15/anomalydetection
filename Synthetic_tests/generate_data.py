from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import roc_auc_score
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize


X_train, y_train, X_test, y_test = generate_data(n_train=100000, n_test=10000, n_features=10, contamination=0.1)

clf_name = 'KNN'
clf=KNN()
clf.fit(X_train)

"""
X_test_reshape = X_test.reshape(1, -1)
print(X_test_reshape)
y_test_pred = clf.predict(X_test_reshape)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test_reshape)  # outlier score
"""

# Predict on the testing data
X_test_reshape = X_test.reshape(-1, 10)  # Reshape testing data to match the number of features
y_test_pred = clf.predict(X_test_reshape)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test_reshape)  # outlier score

roc_score = roc_auc_score(y_test, y_test_scores)

print("Test Predicitons:", y_test_pred)
print("Test Scores:", y_test_scores)
print("ROC-AUC:", roc_score)