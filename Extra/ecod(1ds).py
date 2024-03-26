from pyod.models.ecod import ECOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from joblib import dump, load
from pyod.utils.example import visualize
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import time

start_time = time.time()

X_train, y_train= generate_data(n_train=20000, train_only=True, n_features=2, contamination=0.1, random_state=42)
clf_name = 'ECOD'
clf = ECOD()

clf.fit(X_train)

#X_test = X_test_t.reshape(-1, 5)  # Reshape testing data to match the number of features

y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_

y_test_pred = clf.predict(X_train)
y_test_scores = clf.decision_function(X_train)

roc_score = roc_auc_score(y_train, y_test_scores)

print(evaluate_print(clf_name, y_train, y_train_scores))
#print(evaluate_print(clf_name, y_test, y_test_scores))

#visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, show_figure=True, save_figure=False)

fpr, tpr, _ = roc_curve(y_train, y_test_scores)

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_score, estimator_name='ECOD')
display.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

print('Run time is:', round(time.time() - start_time, 3), 'seconds')