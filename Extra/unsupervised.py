import pandas as pd

from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import SMOTE
from utils.unsupervised_learning import model_ecod, model_copod, model_iforest, model_pca, model_dbscan
from utils.supervised_learning import model_knn, model_nb
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from utils.paramet_tune import IsolationForest_tuner, DBSCAN_tuner
from adjustText import adjust_text
import matplotlib.pyplot as plt


# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/fin_paysys/finpays.csv')

# Encoding categorical values with numerical variables
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

# Determining the X and y values
X = df.drop('fraud', axis=1)
y = df['fraud'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Standardize the data

dbscan_tuner = DBSCAN_tuner(X_scaled, y)
dbscan_cluster = dbscan_tuner.tune_model()
# Save the best parameters
best_eps = dbscan_cluster['eps']
best_min_samples = int(dbscan_cluster['min_samples'])
# Evaluate the DBSCAN model
roc_auc_dbscan, f1_score_dbscan, precision_dbscan, recall_dbscan, accuracy_dbscan, runtime_dbscan = model_dbscan(
    X_scaled, y, eps=best_eps, min_samples=best_min_samples)

# metrics = [
#     {'Model': 'KNN',
#      'Estimator': '3',
#      'ROC_AUC_Score': 0.688,
#      'F1_Score': 0.995,
#      'Runtime': 45.894},
#
#     {'Model': 'Random Forest Classifier',
#      'Estimator': '870',
#      'ROC_AUC_Score': 0.808,
#      'F1_Score': 0.997,
#      'Runtime': 2591.672},
#
#     {'Model': 'XGBoost',
#      'Estimator': '970',
#      'ROC_AUC_Score': 0.906,
#      'F1_Score': 0.994,
#      'Runtime': 32.68},
#
#     {'Model': 'SVM',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.621,
#      'F1_Score': 0.994,
#      'Runtime': 7901.67},
#
#     {'Model': 'Naive Bayes',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.5,
#      'F1_Score': 0.991,
#      'Runtime': 1.876},
#
#     {'Model': 'CatBoost',
#      'Estimator': '580',
#      'ROC_AUC_Score': 0.816,
#      'F1_Score': 0.997,
#      'Runtime': 194.367},
#
#     {'Model': 'LOF',
#      'Estimator': '41',
#      'ROC_AUC_Score': 0.746,
#      'F1_Score': 0.986,
#      'Runtime': 231.333},
#
#     {'Model': 'Principal Component',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.749,
#      'F1_Score': 0.942,
#      'Runtime': 2.335},
#
#     {'Model': 'Isolation Forest',
#      'Estimator': '40',
#      'ROC_AUC_Score': 0.243,
#      'F1_Score': 0.859,
#      'Runtime': 44.325},
#
#     {'Model': 'DBSCAN',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.867,
#      'F1_Score': 0.974,
#      'Runtime': 628.642},
#
#     {'Model': 'COPOD',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.682,
#      'F1_Score': 0.94,
#      'Runtime': 27.019},
#
#     {'Model': 'ECOD',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.646,
#      'F1_Score': 0.94,
#      'Runtime': 25.61}
# ]
#
# metrics_unsupervised = [
#     {'Model': 'LOF',
#      'Estimator': '41',
#      'ROC_AUC_Score': 0.746,
#      'F1_Score': 0.986,
#      'Precision': 0.991,
#      'Recall': 0.981,
#      'Accuracy': 0.639,
#      'Runtime': 231.333},
#
#     {'Model': 'Principal Component',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.749,
#      'F1_Score': 0.942,
#      'Precision': 0.991,
#      'Recall': 0.9,
#      'Accuracy': 0.709,
#      'Runtime': 2.335},
#
#     {'Model': 'Isolation Forest',
#      'Estimator': '440',
#      'ROC_AUC_Score': 0.243,
#      'F1_Score': 0.859,
#      'Precision': 0.992,
#      'Recall': 0.761,
#      'Accuracy': 0.706,
#      'Runtime': 44.325},
#
#     {'Model': 'DBSCAN',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.867,
#      'F1_Score': 0.974,
#      'Precision': 0.993,
#      'Recall': 0.958,
#      'Accuracy': 0.867,
#      'Runtime': 628.642},
#
#     {'Model': 'COPOD',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.682,
#      'F1_Score': 0.94,
#      'Precision': 0.989,
#      'Recall': 0.897,
#      'Accuracy': 0.567,
#      'Runtime': 27.019},
#
#     {'Model': 'ECOD',
#      'Estimator': None,
#      'ROC_AUC_Score': 0.646,
#      'F1_Score': 0.94,
#      'Precision': 0.989,
#      'Recall': 0.897,
#      'Accuracy': 0.551,
#      'Runtime': 25.61}
# ]
#
#
# # Create a dataframe to store the evaluation metrics
# metrics_df = pd.DataFrame(metrics)
# unsupervised_metrics_df = pd.DataFrame(metrics_unsupervised)
#
# # Save the unsupervised metrics to a CSV file
# metrics_df.to_csv('../results/Metrics(DS1).csv', index=False)
#
#
# unsupervised_metrics_df.to_csv('../results/Unsupervised_Metrics(DS1).csv', index=False)
#
#
# # Visualizing the results ROC-AUC Score - Runtime
# plt.figure(figsize=(10, 6))
# plt.scatter(metrics_df['Runtime'], metrics_df['ROC_AUC_Score'], color='blue', s=100)
# texts = []
# for i, txt in enumerate(metrics_df['Model']):
#     texts.append(plt.text(metrics_df['Runtime'][i], metrics_df['ROC_AUC_Score'][i], txt, fontsize=12))
# adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'))
# plt.grid(True)
# plt.xlabel('Runtime', fontsize=14, fontweight='bold')
# plt.ylabel('ROC AUC Score', fontsize=14, fontweight='bold')
# plt.title('ROC AUC Score vs Runtime comparison', fontsize=16, fontweight='bold')
# plt.savefig('../results/ROC_AUC_vs_Runtime(DS1).png', bbox_inches='tight')
# plt.show()
#
# # Visualizing the results F1 Score - Runtime
# plt.figure(figsize=(10, 6))
# plt.scatter(metrics_df['Runtime'], metrics_df['F1_Score'], color='blue', s=100)
# texts = []
# for i, txt in enumerate(metrics_df['Model']):
#     texts.append(plt.text(metrics_df['Runtime'][i], metrics_df['F1_Score'][i], txt, fontsize=12))
# adjust_text(texts=texts, arrowprops=dict(arrowstyle='-', color='grey'))
# plt.grid(True)
# plt.xlabel('Runtime', fontsize=14, fontweight='bold')
# plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
# plt.title('F1 Score vs Runtime comparison', fontsize=16, fontweight='bold')
# plt.savefig('../results/F1_Score_vs_Runtime(DS1).png', bbox_inches='tight')
# plt.show()