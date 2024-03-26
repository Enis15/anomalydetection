from pyod.models.xgbod import XGBOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import time

#datasets_size = [100000, 200000, 30000, 40000, 50000, 60000]
execution_time = []
X_train, X_test, y_train, y_test = generate_data(n_train=500, n_test=500, n_features=2, contamination=0.1, random_state=42)

clf_name = 'XGBOD'
clf = XGBOD(n_components=4,random_state=100)

#for size in datasets_size:
start_time = time.time()

clf.fit(X_train, y_train)

y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_
y_train_scores = clf.decision_function(X_train)

y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

roc_score = roc_auc_score(y_test, y_test_scores)
evaluation = evaluate_print(clf_name, y_test, y_train_scores)
print('Evaluation for dataset size 600:', evaluation)

fpr, tpr, _ = roc_curve(y_test, y_test_scores)

display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_score, estimator_name='XGBOD')
display.plot()
plt.title('Receiver Operating Characteristic (ROC) Curve')
    #plt.show()

execution_time.append(time.time() - start_time)
print('Run time is:', round(time.time() - start_time, 3), 'seconds')

plt.plot('600', execution_time, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Scalability of XGBOD')
plt.grid(True)
plt.show()