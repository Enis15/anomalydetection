import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

metrics_df = pd.read_csv('Metrics(DS4).csv')


# Visualizing the results
plt.figure(figsize=(10, 6))
plt.scatter(metrics_df['Runtime'], metrics_df['F1 Score'], color='green', s=100)
texts = []
for i, txt in enumerate(metrics_df['Model']):
    texts.append(plt.text(metrics_df['Runtime'][i], metrics_df['F1 Score'][i], txt, fontsize=12))
adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'))
plt.grid(True)
plt.xlabel('Runtime', fontsize=14, fontweight='bold')
plt.ylabel('ROC AUC', fontsize=14, fontweight='bold')
plt.title('F1 Score vs Runtime comparison', fontsize=16, fontweight='bold')
plt.savefig('./F1_score_vs_Runtime(DS4).png', bbox_inches='tight')
plt.show()