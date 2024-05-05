from utils.supervised_learning import model_knn
from utils.supervised_learning import model_rf
import pandas as pd
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support
from pyod.utils.data import generate_data
from utils.paramet_tune import paramet_tune

#X_train, X_test, y_train, y_test = generate_data(n_train=10000, n_test=2000, n_features=2, contamination=0.1, random_state=42)

#Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

#Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

if __name__ == '__main__':
    freeze_support()
    best_knn_model = paramet_tune(X_train, y_train, model_name='knn')
    print(best_knn_model)
    k_value = best_knn_model['learner'].n_neighbors

    roc_auc_knn, f1_score_knn, runtime_knn = model_knn(X_train, X_test, y_train, y_test, k_value)

    best_rf_model = paramet_tune(X_train, y_train, model_name='random_forest')
    print(best_rf_model)
    rf_value = best_rf_model['learner'].n_estimators

    roc_auc_rf, f1_score_rf, runtime_rf = model_rf(X_train, X_test, y_train, y_test, rf_value)

    metrics_df = pd.DataFrame({
        'Model': ['KNN', 'Random Forest'],
        'Estimator': [k_value, rf_value],
        'ROC_AUC': [roc_auc_knn, roc_auc_rf],
        'F1_Score': [f1_score_knn, f1_score_rf],
        'Runtime': [runtime_knn, runtime_rf],
    })

    metrics_df.to_csv('Metrics(DS1).csv', index=False)