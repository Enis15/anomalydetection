import matplotlib.pyplot as plt
import pandas as pd

metrics_roc = pd.read_csv('../results/Scalability_test(ROC_AUC).csv')
metrics_f1 = pd.read_csv('../results/Scalability_test(F1_scores).csv')
metrics_time = pd.read_csv('../results/Scalability_test(Runtime).csv')

datasets = [50000, 150000, 350000, 550000, 750000, 1000000]

##Visualize the ROC_AUC scores
plt.figure(figsize=(10, 6))
for model, scores in metrics_roc.items():
    plt.plot(datasets, scores, marker='o', linestyle='--', label=model)
plt.xlabel('Dataset size')
plt.ylabel('ROC AUC Scores')
plt.legend(bbox_to_anchor=(0.92, 0.5), loc='center left')
plt.savefig('../results/Scalability_test(ROC_AUC).png')
plt.show()

# # Visualize the F1 scores
# plt.figure(figsize=(10, 6))
# for model, scores in metrics_f1.items():
#     plt.plot(datasets, scores, marker='o', linestyle='--', label=model)
# plt.xlabel('Dataset size')
# plt.ylabel('F1 Scores')
# plt.legend(loc='best')
# plt.savefig('../results/Scalability_test(F1_scores).png')
# plt.show()
#
# # Visualize the Runtimes scores
# plt.figure(figsize=(10, 6))
# for model, scores in metrics_time.items():
#     plt.plot(datasets, scores, marker='o', linestyle='--', label=model)
# plt.xlabel('Dataset size')
# plt.ylabel('Runtime Scores')
# plt.legend(loc='best')
# plt.savefig('../results/Scalability_test(Runtime).png')
# plt.show()