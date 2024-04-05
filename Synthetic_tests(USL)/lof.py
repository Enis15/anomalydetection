from pyod.models.lof import LOF
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time


#Dataset sizes to evaluate scalability
dataset_sizes = [10000, 20000, 30000, 40000, 50000, 60000]

clf_name = 'LOF'
clf = LOF()

#List of evaluation metrics
execution_times = []
roc_auc_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for size in dataset_sizes:
    print(f'Evaluating dataset size: {size}')

    #Generating synthetic data
    X_train, y_train= generate_data(n_train=size, train_only=True, n_features=2, contamination=0.1, random_state=42)

    # Measure execution time
    start_time = time.time()

    # Fit the model
    clf.fit(X_train)

    y_train_pred = clf.labels_
    y_train_scores = clf.decision_scores_

    # Make predicitons on data
    y_pred = clf.predict(X_train)

    #Runtime evaluation
    execution_times.append(time.time() - start_time)

    # Evalution metrics
    roc_auc_scores.append(roc_auc_score(y_train, y_pred))
    accuracy_scores.append(accuracy_score(y_train, y_pred))
    precision_scores.append(precision_score(y_train, y_pred))
    recall_scores.append(recall_score(y_train, y_pred))
    f1_scores.append(f1_score(y_train, y_pred))

    print('Run time is:', round(time.time() - start_time, 3), 'seconds')

# Plot results
plt.figure(figsize=(12, 6))

# Plot for runtime
plt.subplot(1, 2, 1)
plt.plot(dataset_sizes, execution_times, marker='.', color='teal')
plt.xlabel('Dataset size')
plt.ylabel('Execution Time(seconds)')
plt.title('Dataset size vs Execution time (LOF)')
# plt.savefig('LOF_time.png', bbox_inches='tight')

# Plots for the metrics
plt.subplot(1, 2, 2)
plt.plot(dataset_sizes, roc_auc_scores, marker='.', label='ROC AUC', color='royalblue')
plt.plot(dataset_sizes, accuracy_scores, marker='.', label='Accuracy', color='forestgreen')
plt.plot(dataset_sizes, precision_scores, marker='.', label='Precision', color='firebrick')
plt.plot(dataset_sizes, recall_scores, marker='.', label='Recall', color='orange')
plt.plot(dataset_sizes, f1_scores, marker='.', label='F1 score', color='violet')
plt.xlabel('Dataset size')
plt.ylabel('Score')
plt.title('Dataset size vs Evaluation Metrics (LOF)')
plt.legend()
# plt.savefig('LOF_metrics.png', bbox_inches='tight')

plt.tight_layout()
plt.show()