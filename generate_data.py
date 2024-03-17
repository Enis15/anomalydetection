from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

X_train, y_train, X_test, y_test = generate_data(n_train=200, n_test=100, n_features=2, contamination=0.1)
clf_name = 'KNN'
clf=KNN()
clf.fit(X_train)

y_train_pred = clf.labels_
y_train_score = clf.decision_scores_
print(y_train_pred)

y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores
