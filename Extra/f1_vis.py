import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv('../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv')
df['anomaly'] = df['anomaly'].map({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})
df['anomaly'] = df['anomaly'].astype(int)
data = df[df['anomaly'] == 1]

