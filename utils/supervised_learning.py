from sklearn.metrics import roc_auc_score, f1_score
import time
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

#Define function for KNN (K-Nearest Neighbors) Algorithm for Anomaly Detection
def model_knn(X_train, X_test, y_train, y_test, k):
    """
      KNN Algorithm for anomaly detection.

      Parameters:
          X_train: Input training data, where rows are samples and columns are features.
          X_test: Input test data, where rows are samples and columns are features.
          y_train: Target training data, where rows are samples and columns are labels.
          y_test: Target test data, where rows are samples and columns are labels.
          k: number of neighbors.

      Returns:
          roc_auc_score_knn: ROC AUC score.
          f1_score_knn: F1 score.
          runtime_knn: Runtime of KNN algorithm.
      """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = KNeighborsClassifier(n_neighbors=k, metric='minkowski', n_jobs=-1)
    model.fit(X_train, y_train)

    #Get the prediction labels and scores for the test data
    y_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)

    #Evaluation metrics
    roc_auc_knn = round(roc_auc_score(y_test, y_pred), 3)
    f1_score_knn = round(f1_score(y_test, y_pred, average='weighted'), 3)
    runtime_knn = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for KNN model, with k = {k}, are: \n'
          f'ROC AUC: {roc_auc_knn}\n'
          f'F1 score: {f1_score_knn}\n' 
          f'Time elapsed: {runtime_knn}')

    return roc_auc_knn, f1_score_knn, runtime_knn

#Define function for XGBOOST Algorithm for Anomaly Detection
def model_xgboost(X_train, X_test, y_train, y_test, n_estimators, max_depth):
    """
      XGBoost Algorithm for anomaly detection.

      Parameters:
          X_train: Input training data, where rows are samples and columns are features.
          X_test: testing Input test data, where rows are samples and columns are features.
          y_train: Target training data, where rows are samples and columns are labels.
          y_test: Target test data, where rows are samples and columns are labels.
          n_estimators: number of estimators.

      Returns:
          roc_auc_score_xgboost: ROC AUC score.
          f1_score_xgboost: F1 score.
          runtime_xgboost: Runtime of XBoost Algorithm.
      """
    # Record the start time
    start_time = time.time()

    #Create a dictonary with the parameters needed to initate the classifier
    params = {
        'objective': 'binary:logistic',
        'max_depth': max_depth,
        'learning_rate': 0.05,
        'n_estimators': n_estimators,
    }

    # Define the model and the parameters
    model = XGBClassifier()
    model.fit(X_train, y_train)

     # Get the prediction labels and scores for the test data
    y_pred = model.predict(X_test)  # Outlier labels (1 = outliers & 0 = inliers)

    # Evaluation metrics
    roc_auc_xgboost = round(roc_auc_score(y_test, y_pred), 3)
    f1_score_xgboost = round(f1_score(y_test, y_pred, average='weighted'), 3)
    runtime_xgboost = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for XGBoost model, are: \n'
          f'ROC AUC: {roc_auc_xgboost}\n'
          f'F1 score: {f1_score_xgboost}\n'
          f'Time elapsed: {runtime_xgboost}')

    return roc_auc_xgboost, f1_score_xgboost, runtime_xgboost

#Define function for SVM (Support Vector Machine) Algorithm for Anomaly Detection
def model_svm(X_train, X_test, y_train, y_test):
    """
      SVM Algorithm for anomaly detection.

      Parameters:
          X_train: Input training data, where rows are samples and columns are features.
          X_test: Input test data, where rows are samples and columns are features.
          y_train: Target training data, where rows are samples and columns are labels.
          y_test: Target test data, where rows are samples and columns are labels.


      Returns:
          roc_auc_score_knn: ROC AUC score.
          f1_score_knn: F1 score.
          runtime_knn: Runtime of SVM Algorithm.
      """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = svm.SVC()
    model.fit(X_train, y_train)

    #Get the prediction labels and scores for the test data
    y_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)

    #Evaluation metrics
    roc_auc_svm = round(roc_auc_score(y_test, y_pred), 3)
    f1_score_svm = round(f1_score(y_test, y_pred, average='weighted'), 3)
    runtime_svm = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for SVM model, are: \n'
          f'ROC AUC: {roc_auc_svm}\n'
          f'F1 score: {f1_score_svm}\n' 
          f'Time elapsed: {runtime_svm}')

    return roc_auc_svm, f1_score_svm, runtime_svm

#Define function for Naive Bayes Algorithm
def model_nb(X_train, X_test, y_train, y_test):
    """
    SVM Algorithm for anomaly detection.

    Parameters:
        X_train: Input training data, where rows are samples and columns are features.
        X_test: Input test data, where rows are samples and columns are features.
        y_train: Target training data, where rows are samples and columns are labels.
        y_test: Target test data, where rows are samples and columns are labels.


    Returns:
        roc_auc_score_svm: ROC AUC score.
        f1_score_svm: F1 score.
        runtime_svm: Runtime of Naive Bayes Algorithm.
    """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = GaussianNB()
    model.fit(X_train, y_train)

    #Get the prediction labels and scores for the test data
    y_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)

    #Evaluation metrics
    roc_auc_nb = round(roc_auc_score(y_test, y_pred), 3)
    f1_score_nb = round(f1_score(y_test, y_pred, average='weighted'), 3)
    runtime_nb = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Naive Bayes model, are: \n'
          f'ROC AUC: {roc_auc_nb}\n'
          f'F1 score: {f1_score_nb}\n' 
          f'Time elapsed: {runtime_nb}')

    return roc_auc_nb, f1_score_nb, runtime_nb

#Define function for Random Forest Algorithm
def model_rf(X_train, X_test, y_train, y_test, k):
    """
        Random Forest Algorithm for anomaly detection.

        Parameters:
            X_train: Input training data, where rows are samples and columns are features.
            X_test: Input test data, where rows are samples and columns are features.
            y_train: Target training data, where rows are samples and columns are labels.
            y_test: Target test data, where rows are samples and columns are labels.
            k: The number of estimators.

        Returns:
            roc_auc_score_rf: ROC AUC score.
            f1_score_rf: F1 score.
            runtime_rf: Runtime of Random Forest Algorithm.
        """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = RandomForestClassifier(n_estimators=k, n_jobs=-1)
    model.fit(X_train, y_train)

    #Get the prediction labels and scores for the test data
    y_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)

    #Evaluation metrics
    roc_auc_rf = round(roc_auc_score(y_test, y_pred), 3)
    f1_score_rf = round(f1_score(y_test, y_pred, average='weighted'), 3)
    runtime_rf = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Random Forest model, are: \n'
          f'ROC AUC: {roc_auc_rf}\n'
          f'F1 score: {f1_score_rf}\n' 
          f'Time elapsed: {runtime_rf}')

    return roc_auc_rf, f1_score_rf, runtime_rf

#Define function for CatBoost Algorithm
def model_cb(X_train, X_test, y_train, y_test, iterations, learning_rate, depth):
    """
        CatBoost Algorithm for anomaly detection.

        Parameters:
            X_train: Input training data, where rows are samples and columns are features.
            X_test: Input test data, where rows are samples and columns are features.
            y_train: Target training data, where rows are samples and columns are labels.
            y_test: Target test data, where rows are samples and columns are labels.


        Returns:
            roc_auc_score_cb: ROC AUC score.
            f1_score_cb: F1 score.
            runtime_cb: Runtime of CatBoost Algorithm.
        """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = CatBoostClassifier(iterations=iterations,
                               learning_rate=learning_rate,
                               depth=depth,
                               verbose=False)

    model.fit(X_train, y_train)

    #Get the prediction labels and scores for the test data
    y_pred = model.predict(X_test)  #Outlier labels (1 = outliers & 0 = inliers)

    #Evaluation metrics
    roc_auc_cb = round(roc_auc_score(y_test, y_pred), 3)
    f1_score_cb = round(f1_score(y_test, y_pred, average='weighted'), 3)
    runtime_cb = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for CatBoost model are: \n'
          f'ROC AUC: {roc_auc_cb}\n'
          f'F1 score: {f1_score_cb}\n' 
          f'Time elapsed: {runtime_cb}')

    return roc_auc_cb, f1_score_cb, runtime_cb