import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# df = pd.read_csv('../results/nearest_dist/dataset4_dist_summary.csv.csv')
#
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='avg_centroid_dist', y='avg_anomaly_distance', s=100, alpha=0.8, data=df, edgecolor='white')
# plt.title('Avg comparison')
# plt.xlabel('Centroid distance')
# plt.ylabel('Anomaly distance')
# plt.grid(True)
# plt.show()

df = pd.read_csv('../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv')
print(df.shape)