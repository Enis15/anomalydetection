import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn, model_rf, model_nb, model_cb, model_svm, model_xgboost
from utils.unsupervised_learning import model_lof, model_pca, model_iforest, model_ecod, model_copod, model_kmeans
import matplotlib.pyplot as plt
from math import sqrt

datasets = [50000, 150000, 350000, 550000, 750000, 1000000]

kmeans_val = {
    'ROC_AUC': [],
    'F1_score': [],
    'Runtime': []
}

lof_val = {
    'ROC_AUC': [],
    'F1_score': [],
    'Runtime': []
}


for dataset in datasets:

    X, y = make_classification(n_samples=dataset, n_features=15, n_classes=2, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # MODEL LOCAL OUTLIER FACTOR (LOF)

    # Evaluate the LOF model
    roc_auc_lof, f1_score_lof, runtime_lof = model_lof(X, y, 20) # Using default value of n_neighbors=20

    lof_val['ROC_AUC'].append(roc_auc_lof)
    lof_val['F1_score'].append(f1_score_lof)
    lof_val['Runtime'].append(runtime_lof)

    # MODEL K - MEANS

    # Evaluate the K-means model
    roc_auc_kmeans, f1_score_kmeans, runtime_kmeans = model_kmeans(X, y, 8)  # Using default value of k=8

    kmeans_val['ROC_AUC'].append(roc_auc_kmeans)
    kmeans_val['F1_score'].append(f1_score_kmeans)
    kmeans_val['Runtime'].append(runtime_kmeans)
