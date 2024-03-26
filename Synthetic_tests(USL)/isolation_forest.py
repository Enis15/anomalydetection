from pyod.models.iforest import IForest
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import time

datasets_size = [100000, 200000, 30000, 40000, 50000, 60000]
execution_time = []

clf_name = 'IForest'
clf = IForest()

for size in datasets_size:
    X_train, y_train= generate_data(n_train=size, train_only=True, n_features=2, contamination=0.1, random_state=42)

    start_time = time.time()

    clf.fit(X_train)

    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    y_test_pred = clf.predict(X_train)
    y_test_scores = clf.decision_function(X_train)

    roc_score = roc_auc_score(y_train, y_test_scores)
    evaluation = evaluate_print(clf_name, y_train, y_train_scores)
    print('Evaluation for dataset size', size, ':', evaluation)

    fpr, tpr, _ = roc_curve(y_train, y_test_scores)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_score, estimator_name='IForest')
    display.plot()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    #plt.show()

    execution_time.append(time.time() - start_time)
    print('Run time is:', round(time.time() - start_time, 3), 'seconds')

plt.plot(datasets_size, execution_time, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Scalability of IForest')
plt.grid(True)
plt.show()