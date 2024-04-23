from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score
import time
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD

#Define function for LOF (Local Outlier Factor) Algorithm
def model_lof(X, y, k):
    """
    LOF Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        k: number of neighbors.

    Returns:
        roc_auc_score_lof: ROC AUC score.
        f1_score_lof: F1 score.
        runtime_lof: Runtime of LOF.
    """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = LOF(n_neighbors=k)
    model.fit(X)

    #Get the prediction labels and scores for the test data
    X_labels = model.predict(X)  #Outlier labels (1 = outliers & 0 = inliers)
    X_scores = model.decision_function(X) #The raw outlier scores

    #Evaluation metrics
    roc_auc_lof = roc_auc_score(y, X_labels)
    f1_score_lof = f1_score(y, X_labels, average='weighted')
    runtime_lof = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for LOF model, with n_neighbors = {k}, are: \n'
          f'ROC AUC: {roc_auc_lof}\n'
          f'F1 score: {f1_score_lof}\n' 
          f'Time elapsed: {runtime_lof}')

    return roc_auc_lof, f1_score_lof, runtime_lof

#Define function for CBLOF (Cluster Based Local Outlier Factor) Algorithm
def model_cblof(X, y, k):
    """
    CBLOF Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        k: number of neighbors.

    Returns:
        roc_auc_score_cblof: ROC AUC score.
        f1_score_cblof: F1 score.
        runtime_cblof: Runtime of LOF.

  """
    #Record the start time
    start_time = time.time()

    #Define the model and the parameters
    model = CBLOF(n_clusters=k)
    model.fit(X)

    #Get the prediction labels and scores for the test data
    X_labels = model.predict(X)  #Outlier labels (1 = outliers & 0 = inliers)
    X_scores = model.decision_function(X) #The raw outlier scores

    #Evaluation metrics
    roc_auc_cblof = roc_auc_score(y, X_labels)
    f1_score_cblof = f1_score(y, X_labels, average='weighted')
    runtime_cblof = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for CBLOF model, with n_clusters = {k}, are: \n'
          f'ROC AUC: {roc_auc_cblof}\n'
          f'F1 score: {f1_score_cblof}\n' 
          f'Time elapsed: {runtime_cblof}')

    return roc_auc_cblof, f1_score_cblof, runtime_cblof


# Define function for IForest (Isolation Forest) Algorithm
def model_iforest(X, y, k):
    """
    Isolation Forest Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        k: number of neighbors.

    Returns:
        roc_auc_score_iforest: ROC AUC score.
        f1_score_iforest: F1 score.
        runtime_iforest: Runtime of LOF.

  """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = IForest(n_estimators=k)
    model.fit(X)

    # Get the prediction labels and scores for the test data
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliers)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_iforest = roc_auc_score(y, X_labels)
    f1_score_iforest = f1_score(y, X_labels, average='weighted')
    runtime_iforest= round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Isolation Forest model, with n_estimators = {k}, are: \n'
          f'ROC AUC: {roc_auc_iforest}\n'
          f'F1 score: {f1_score_iforest}\n'
          f'Time elapsed: {runtime_iforest}')

    return roc_auc_iforest, f1_score_iforest, runtime_iforest

# Define function for PCA (Principal Component Analysis) Algorithm
def model_pca(X, y, k):
    """
    Isolation PCA Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        k: number of neighbors.

    Returns:
        roc_auc_score_pca: ROC AUC score.
        f1_score_pca: F1 score.
        runtime_pca: Runtime of LOF.

  """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = PCA()
    model.fit(X)

    # Get the prediction labels and scores for the test data
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliers)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_pca = roc_auc_score(y, X_labels)
    f1_score_pca = f1_score(y, X_labels, average='weighted')
    runtime_pca = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for PCA model are: \n'
          f'ROC AUC: {roc_auc_pca}\n'
          f'F1 score: {f1_score_pca}\n'
          f'Time elapsed: {runtime_pca}')

    return roc_auc_pca, f1_score_pca, runtime_pca

# Define function for COPOD (Copula-Based Outlier Detection) Algorithm
def model_copod(X, y, k):
    """
    Isolation Forest Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        k: number of neighbors.

    Returns:
        roc_auc_score_copod: ROC AUC score.
        f1_score_copod: F1 score.
        runtime_copod: Runtime of LOF.

  """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = COPOD()
    model.fit(X)

    # Get the prediction labels and scores for the test data
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliers)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_copod = roc_auc_score(y, X_labels)
    f1_score_copod = f1_score(y, X_labels, average='weighted')
    runtime_copod = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for COPOD model are: \n'
          f'ROC AUC: {roc_auc_copod}\n'
          f'F1 score: {f1_score_copod}\n'
          f'Time elapsed: {runtime_copod}')

    return roc_auc_copod, f1_score_copod, runtime_copod

# Define function for ECOD (Empirical Cumulative Outlier Detection) Algorithm
def model_ecod(X, y, k):
    """
    Isolation Forest Algorithm for anomaly detection.

    Parameters:
        X: Input dataframe, where rows are samples and columns are features.
        y: True labels, used to for evaluation metrics
        k: number of neighbors.

    Returns:
        roc_auc_score_ecod: ROC AUC score.
        f1_score_ecod: F1 score.
        runtime_ecod: Runtime of LOF.

  """
    # Record the start time
    start_time = time.time()

    # Define the model and the parameters
    model = ECOD()
    model.fit(X)

    # Get the prediction labels and scores for the test data
    X_labels = model.predict(X)  # Outlier labels (1 = outliers & 0 = inliers)
    X_scores = model.decision_function(X)  # The raw outlier scores

    # Evaluation metrics
    roc_auc_ecod = roc_auc_score(y, X_labels)
    f1_score_ecod = f1_score(y, X_labels, average='weighted')
    runtime_ecod = round(time.time() - start_time, 3)

    print(f'Evaluation metrics for Isolation Forest model are: \n'
          f'ROC AUC: {roc_auc_ecod}\n'
          f'F1 score: {f1_score_ecod}\n'
          f'Time elapsed: {runtime_ecod}')

    return roc_auc_ecod, f1_score_ecod, runtime_ecod