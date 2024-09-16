import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text

# #metrics_df = pd.read_csv('Scalability_test(F1_scores).csv')
#
# datasets = [50000, 150000, 350000, 550000, 750000, 1000000]
#
#
# f1_scores = {
#     'KNN': [0.947, 0.908, 0.84, 0.908, 0.928, 0.96],
#     'Random Forest': [0.977, 0.934, 0.871, 0.923, 0.973, 0.981],
#     'XGBoost': [0.977, 0.933, 0.871, 0.923, 0.973, 0.98],
#     'SVM': [0.97, 0.921, 0.853, 0.92, 0.958, 0.975],
#     'Naive Bayes': [0.909, 0.902, 0.852, 0.913, 0.871, 0.91],
#     'CATboost': [0.976, 0.93, 0.867, 0.922, 0.973, 0.98],
#     'LOF': [0.333, 0.334, 0.333, 0.333, 0.333, 0.333],
#     'PCA': [0.456, 0.395, 0.388, 0.393, 0.386, 0.423],
#     'Isolation Forest': [0.468, 0.397, 0.411, 0.375, 0.363, 0.416],
#     'K-Means': [0.246, 0.105, 0.203, 0.017, 0.423, 0.002],
#     'COPOD': [0.471, 0.354, 0.343, 0.345, 0.353, 0.445],
#     'ECOD': [0.44, 0.402, 0.394, 0.402, 0.413, 0.419]
# }
#
# execution_times = {
#     'KNN': [3.761, 19.892, 135.869, 174.492, 227.831, 375.729],
#     'Random Forest': [2.748, 9.949, 57.918, 68.143, 102.089, 148.747],
#     'XGBoost': [0.368, 0.459, 2.355, 1.942, 2.014, 2.661],
#     'SVM': [9.992, 198.066, 2027.98, 2446.787, 3233.229, 3042.688],
#     'Naive Bayes': [0.016, 0.09, 0.119, 0.188, 0.244, 0.314],
#     'CATboost': [0.46, 0.717, 1.553, 2.313, 3.179, 4.121],
#     'LOF': [24.371, 237.681, 831.512, 1505.998, 2345.019, 3550.705],
#     'PCA': [0.082, 0.218, 0.767, 0.705, 1.019, 1.303],
#     'Isolation Forest': [0.816, 3.497, 5.362, 7.32, 10.063, 13.53],
#     'K-Means': [0.607, 2.321, 4.969, 6.373, 8.884, 10.274],
#     'COPOD': [1.182, 5.137, 7.715, 12.273, 19.725, 24.604],
#     'ECOD': [0.85, 4.501, 7.596, 12.473, 17.621, 23.84]
# }

metrics_df = pd.read_csv("../results/Metrics(DS3).csv")
# Visualizing the results ROC-AUC Score - Runtime
plt.figure(figsize=(10, 6))
plt.scatter(metrics_df['Runtime'], metrics_df['ROC_AUC_Score'], color='blue', s=100)
texts = []
for i, txt in enumerate(metrics_df['Model']):
    texts.append(plt.text(metrics_df['Runtime'][i], metrics_df['ROC_AUC_Score'][i], txt, fontsize=12))
adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'))
plt.grid(True)
plt.xlabel('Runtime', fontsize=14, fontweight='bold')
plt.ylabel('ROC AUC Score', fontsize=14, fontweight='bold')
plt.title('ROC AUC Score vs Runtime comparison', fontsize=16, fontweight='bold')
plt.savefig('../results/ROC_AUC_vs_Runtime(DS3).png', bbox_inches='tight')
plt.show()

# Visualizing the results F1 Score - Runtime
plt.figure(figsize=(10, 6))
plt.scatter(metrics_df['Runtime'], metrics_df['F1_Score'], color='blue', s=100)
texts = []
for i, txt in enumerate(metrics_df['Model']):
    texts.append(plt.text(metrics_df['Runtime'][i], metrics_df['F1_Score'][i], txt, fontsize=12))
adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'))
plt.grid(True)
plt.xlabel('Runtime', fontsize=14, fontweight='bold')
plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
plt.title('F1 Score vs Runtime comparison', fontsize=16, fontweight='bold')
plt.savefig('../results/F1_Score_vs_Runtime(DS3).png', bbox_inches='tight')
plt.show()