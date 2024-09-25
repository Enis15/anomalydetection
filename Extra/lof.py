import os
import time
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import gen_batches
import matplotlib.pyplot as plt
from adjustText import adjust_text


metrics_df = pd.read_csv('../results/Metrics(DS4).csv')

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
plt.savefig('../results/ROC_AUC_vs_Runtime(DS4).png', bbox_inches='tight')
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
plt.savefig('../results/F1_Score_vs_Runtime(DS4).png', bbox_inches='tight')
plt.show()
