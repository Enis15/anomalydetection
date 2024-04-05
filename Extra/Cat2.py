from pyod.utils.data import generate_data
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import time

# Define dataset sizes to evaluate scalability
dataset_sizes = [10000, 20000, 30000, 40000, 50000, 60000]

# Initialize CatBoost classifier
clf = CatBoostClassifier(iterations=5, learning_rate=0.1)

# Store execution times and evaluation metrics
execution_times = []
roc_auc_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Define fixed number of testing samples
n_test_samples = 3000

# Iterate over different dataset sizes
for size in dataset_sizes:
    print(f"Evaluating dataset size: {size}")

    # Generate synthetic data
    X_train, _, y_train, _ = generate_data(n_train=size, n_test=(size*0.3), n_features=2, contamination=0.1,
                                           random_state=42)

    # Measure execution time
    start_time = time.time()

    # Fit the model
    clf.fit(X_train, y_train)

    # Record execution time
    execution_times.append(time.time() - start_time)

    # Perform cross-validation for evaluation metrics
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')
    roc_auc_scores.append(cv_scores.mean())

    cv_accuracy_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_scores.append(cv_accuracy_scores.mean())

    cv_precision_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='precision')
    precision_scores.append(cv_precision_scores.mean())

    cv_recall_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='recall')
    recall_scores.append(cv_recall_scores.mean())

    cv_f1_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
    f1_scores.append(cv_f1_scores.mean())

    print('Run time is:', round(time.time() - start_time, 3), 'seconds')

# Plot scalability results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(dataset_sizes, execution_times, marker='o')
plt.xlabel('Dataset Size')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Dataset Size')
plt.grid(True)
# plt.savefig('Catboost_ds.png', bbox_inches='tight')

plt.subplot(1, 2, 2)
plt.plot(dataset_sizes, roc_auc_scores, marker='o', label='ROC-AUC')
plt.plot(dataset_sizes, accuracy_scores, marker='o', label='Accuracy')
plt.plot(dataset_sizes, precision_scores, marker='o', label='Precision')
plt.plot(dataset_sizes, recall_scores, marker='o', label='Recall')
plt.plot(dataset_sizes, f1_scores, marker='o', label='F1')
plt.xlabel('Dataset Size')
plt.ylabel('Score')
plt.title('Evaluation Metrics vs Dataset Size')
plt.legend()
plt.grid(True)
# plt.savefig('Catboost_metrics.png', bbox_inches='tight')


plt.tight_layout()
plt.show()