import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Import the logger function
from utils.logger import logger

# Initialize the logger
_logger = logger(__name__)

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/Fraud.csv')

# Feature engineering: Dropping the columns 'nameOrig' & 'nameDest'; Encoding values to the column 'CASH_OUT'
df = df.drop(['nameOrig', 'nameDest'], axis=1)
df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

features = df.drop(columns=['isFraud'])
scale = StandardScaler()
features_scaled = scale.fit_transform(features)

tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features_scaled)

df['TSNE1'] = tsne_results[:, 0]
df['TSNE2'] = tsne_results[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='isFraud', data=df, palette={0: 'blue', 1: 'red'})
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.title('t-SNE Plot for Dataset 2: Normal & Anomaly points')
plt.savefig('../results/dataset3_anomaly(scaled).png')
plt.show()

tsne_2 = TSNE(n_components=2, random_state=42)
tsne_result = tsne_2.fit_transform(features)

df['TSNE1'] = tsne_result[:, 0]
df['TSNE2'] = tsne_result[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='isFraud', data=df, palette={0: 'blue', 1: 'red'})
plt.xlabel('TSNE 1')
plt.ylabel('TSNE 2')
plt.title('t-SNE Plot for Dataset 2: Normal & Anomaly points')
plt.savefig('../results/dataset3_anomaly.png')
plt.show()