import pandas as pd

filepath = {
    'dataset1': '../results/nearest_dist/dataset1_cluster.csv',
    'dataset2': '../results/nearest_dist/dataset2_cluster.csv',
    'dataset3': '../results/nearest_dist/dataset3_cluster.csv',
    'dataset4': '../results/nearest_dist/dataset4_cluster.csv'
}
dataset_name = 'dataset4'  # Adjust to your desired dataset from the dict
file_path = filepath[dataset_name]

df = pd.read_csv(file_path)

summary_results = df.groupby('Cluster Index').agg(
    avg_anomaly_distance = ('Distance to Centroid', 'mean'),
    avg_centroid_dist = ('Average Distance to Centroid', 'mean'),
    min_anomaly_distance = ('Distance to Centroid', 'min'),
    max_anomaly_distance = ('Distance to Centroid', 'max'),
).reset_index()

# Calculate the ratio between avg_anomaly_distance and avg_centroid_dist to check the distances
summary_results['ratio'] = (summary_results['avg_anomaly_distance'] / summary_results['avg_centroid_dist'])

# Reorder the columns
summary_results = summary_results[
    ['Cluster Index', 'avg_anomaly_distance', 'avg_centroid_dist', 'ratio', 'min_anomaly_distance', 'max_anomaly_distance']
]

summary_csv = summary_results.to_csv(f'../results/nearest_dist/{dataset_name}_dist_summary.csv', index=False)
print('Results saved in csv!')