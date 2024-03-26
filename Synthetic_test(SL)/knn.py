from pyod.models import knn
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.models.knn import KNN
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve, roc_auc_score
import time
from joblib import dump, load

datasets_size = [10000, 20000, 30000, 40000, 50000, 60000]
execution_time = []

clf_name = 'KNN'
clf = KNN() #K number set to 5 by default
for size in datasets_size:
    X_train, X_test, y_train, y_test = generate_data(n_train=size, n_test=(size*0.3), n_features=2, contamination=0.1, random_state=42)

    start_time = time.time()

    clf.fit(X_train, y_train)

    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    y_test_pred = clf.predict(X_test)
    y_test_scores = clf.decision_function(X_test)

    roc_score = roc_auc_score(y_test, y_test_pred)
    evaluation = evaluate_print(clf_name, y_test, y_test_pred)
    print('Evaluation for dataset size', size, ':', evaluation)

    fpr, tpr, _ = roc_curve(y_test, y_test_pred)

    display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_score, estimator_name='KNN')
    display.plot()
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    #plt.show()

    execution_time.append(time.time() - start_time)
    print('Run time is:', round(time.time() - start_time, 3), 'seconds')

plt.plot(datasets_size, execution_time, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (seconds)')
plt.title('KNN')
plt.grid(True)
plt.show()