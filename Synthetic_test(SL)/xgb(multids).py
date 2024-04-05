from pyod.models.xgbod import XGBOD
from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time


#Dataset sizes to evaluate scalability
dataset_sizes = [10000, 20000, 30000, 40000, 50000, 60000]

clf_name = 'XGBOD'

clf = XGBOD(n_components=4,random_state=100)

#List of evaluation metrics
execution_times = []
roc_auc_scores = []
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for size in dataset_sizes:
    
    #Generating synthetic data
    X_train, X_test, y_train, y_test = generate_data(n_train=size, n_test=(size*0.3), n_features=2, contamination=0.1, random_state=42)

    #Measure execution time
    start_time = time.time()

    #Fit the model
    clf.fit(X_train, y_train)

    #Make predicitons on test data
    y_pred = clf.predict(X_test)

    # Runtime evaluation
    execution_times.append(round(time.time() - start_time, 3))

    #Evalution metrics
    roc_auc_scores.append(roc_auc_score(y_test, y_pred))
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

#Printing the results for the evaluation metrics
for i, size in enumerate(dataset_sizes):
    print(f'Evaluation metrics for dataset size {size} are: \n'
      f'ROC_AUC: {roc_auc_scores[i]}\n'
      f'Accuracy: {accuracy_scores[i]}\n'
      f'Precision: {precision_scores[i]}\n'
      f'Recall: {recall_scores[i]}\n'
      f'F1: {f1_scores[i]}\n'
      f'Execution time: {execution_times[i]}')

#Plot results
plt.figure(figsize=(12, 6))

#Plot for runtime
plt.subplot(1, 2, 1)
plt.plot(dataset_sizes, execution_times, marker='.', color='teal')
plt.xlabel('Dataset size')
plt.ylabel('Execution Time(seconds)')
plt.title('Dataset size vs Execution time (XGBoost)')
# plt.savefig('XGboost_time.png', bbox_inches='tight')

#Plots for the metrics
plt.subplot(1, 2, 2)
plt.plot(dataset_sizes, roc_auc_scores, marker='.', label='ROC AUC', color='royalblue')
plt.plot(dataset_sizes, accuracy_scores, marker='.', label='Accuracy', color='forestgreen')
plt.plot(dataset_sizes, precision_scores, marker='.', label='Precision', color='firebrick')
plt.plot(dataset_sizes, recall_scores, marker='.', label='Recall', color='orange')
plt.plot(dataset_sizes, f1_scores, marker='.', label='F1 score', color='violet')
plt.xlabel('Dataset size')
plt.ylabel('Score')
plt.title('Dataset size vs Evaluation Metrics (XGBoost)')
plt.legend()
# plt.savefig('XGBoost_metrics.png', bbox_inches='tight')

plt.tight_layout()
plt.show()