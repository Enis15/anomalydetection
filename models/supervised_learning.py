from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
import time
from pyod.models.knn import KNN
from pyod.models.xgbod import XGBOD
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

#Define function for KNN Algorithm
def model_knn(X_train, X_test, y_train, y_test, k):
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = KNN(n_neighbors=k, contamination=0.1)
    model.fit(X_train, y_train)

    #Get the prediction lables and scores for the training data
    y_train_pred = model.labels_ #Outlier labels (1 = outliers & 0 = inliers)
    y_train_scores = model.decision_scores_ #The raw outlier scores

    #Get the prediction labels and scores for the test data
    y_test_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)
    y_test_scores = model.decision_function(X_test) #The raw outlier scores

    #Evaluation metrics
    roc_auc_knn = roc_auc_score(y_test, y_test_pred)
    f1_score_knn = f1_score(y_test, y_test_pred, average='weighted')
    end_time_knn = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for KNN model, with k = {k}, are: \n'
          f'ROC AUC: {roc_auc_knn}\n'
          f'F1 score: {f1_score_knn}\n' 
          f'Time elapsed: {end_time_knn}')

    return roc_auc_knn, f1_score_knn, end_time_knn

#Define function for XGBOOST Algorithm
def model_xgboost(X_train, X_test, y_train, y_test, k):
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = XGBOD(n_estimators=k, n_jobs=-1)
    model.fit(X_train, y_train)

    # Get the prediction lables and scores for the training data
    y_train_pred = model.labels_  # Outlier labels (1 = outliers & 0 = inliers)
    y_train_scores = model.decision_scores_  # The raw outlier scores

    # Get the prediction labels and scores for the test data
    y_test_pred = model.predict(X_test)  # Outlier labels (1 = outliers & 0 = inliers)
    y_test_scores = model.decision_function(X_test)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_xgboost = roc_auc_score(y_test, y_test_pred)
    f1_score_xgboost = f1_score(y_test, y_test_pred, average='weighted')
    end_time_xgboost = round(time.time() - start_time, 3)

    return roc_auc_xgboost, f1_score_xgboost, end_time_xgboost

#Define function for SVM Algorithm
def model_svm(X_train, X_test, y_train, y_test):
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = svm.SVC()
    model.fit(X_train, y_train)

    #Get the prediction lables and scores for the training data(SVC doesn't support .labels)
    #y_train_pred = model.labels_ #Outlier labels (1 = outliers & 0 = inliers)
    #y_train_scores = model.decision_scores_ #The raw outlier scores

    #Get the prediction labels and scores for the test data
    y_test_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)
    y_test_scores = model.decision_function(X_test) #The raw outlier scores

    #Evaluation metrics
    roc_auc_svm = roc_auc_score(y_test, y_test_pred)
    f1_score_svm = f1_score(y_test, y_test_pred, average='weighted')
    end_time_svm = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for SVM model, are: \n'
          f'ROC AUC: {roc_auc_svm}\n'
          f'F1 score: {f1_score_svm}\n' 
          f'Time elapsed: {end_time_svm}')

    return roc_auc_svm, f1_score_svm, end_time_svm

#Define function for Naive Bayes Algorithm
def model_nb(X_train, X_test, y_train, y_test):
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = GaussianNB()
    model.fit(X_train, y_train)

    #Get the prediction lables and scores for the training data(NB doesn't support .labels)
    #y_train_pred = model.labels_ #Outlier labels (1 = outliers & 0 = inliers)
    #y_train_scores = model.decision_scores_ #The raw outlier scores

    #Get the prediction labels and scores for the test data
    y_test_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)
    #y_test_scores = model.decision_function(X_test) #The raw outlier scores

    #Evaluation metrics
    roc_auc_nb = roc_auc_score(y_test, y_test_pred)
    f1_score_nb = f1_score(y_test, y_test_pred, average='weighted')
    end_time_nb = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Naive Bayes model, are: \n'
          f'ROC AUC: {roc_auc_nb}\n'
          f'F1 score: {f1_score_nb}\n' 
          f'Time elapsed: {end_time_nb}')

    return roc_auc_nb, f1_score_nb, end_time_nb

#Define function for Random Forest Algorithm
def model_rf(X_train, X_test, y_train, y_test, k):
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = RandomForestClassifier(n_estimators=k, n_jobs=-1, random_state = 42)
    model.fit(X_train, y_train)

    #Get the prediction lables and scores for the training data
    #y_train_pred = model.labels_ #Outlier labels (1 = outliers & 0 = inliers)
    #y_train_scores = model.decision_scores_ #The raw outlier scores

    #Get the prediction labels and scores for the test data
    y_test_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)
    #y_test_scores = model.decision_function(X_test) #The raw outlier scores

    #Evaluation metrics
    roc_auc_rf = roc_auc_score(y_test, y_test_pred)
    f1_score_rf = f1_score(y_test, y_test_pred, average='weighted')
    end_time_rf = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Naive Bayes model, are: \n'
          f'ROC AUC: {roc_auc_rf}\n'
          f'F1 score: {f1_score_rf}\n' 
          f'Time elapsed: {end_time_rf}')

    return roc_auc_rf, f1_score_rf, end_time_rf

#Define function for Catboost Algorithm
def model_cb(X_train, X_test, y_train, y_test, k):
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = CatBoostClassifier(iterations = k, learning_rate = 0.1)
    model.fit(X_train, y_train)

    #Get the prediction lables and scores for the training data
    #y_train_pred = model.labels_ #Outlier labels (1 = outliers & 0 = inliers)
    #y_train_scores = model.decision_scores_ #The raw outlier scores

    #Get the prediction labels and scores for the test data
    y_test_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)
    #y_test_scores = model.decision_function(X_test) #The raw outlier scores

    #Evaluation metrics
    roc_auc_cb = roc_auc_score(y_test, y_test_pred)
    f1_score_cb = f1_score(y_test, y_test_pred, average='weighted')
    end_time_cb = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Naive Bayes model, are: \n'
          f'ROC AUC: {roc_auc_cb}\n'
          f'F1 score: {f1_score_cb}\n' 
          f'Time elapsed: {end_time_cb}')

    return roc_auc_cb, f1_score_cb, end_time_cb