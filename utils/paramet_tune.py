import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score, make_scorer, f1_score
from sklearn.model_selection import cross_val_score, KFold

# Import the models
from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier



'''
=======================================
Isolation Forest parameter optimization
=======================================
'''

class IsolationForest_tuner:
    """
    Hyperparameter tuning for Isolation Forest classifier.
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.
    Examples usage:
    if_tuner=IsolationForest_tuner(X, y)
    best_n_estimators = if_tuner.tune_model() --> Used to train the final Isolation Forest model with optimal parameters.
    """
    def __init__(self, X, y):
        """
        Initialize the hyperparameter tuning object.
        Parameters:
            X (array-like): Training data features.
            y (array-like): Ground truth labels.
        """
        self.X = X
        self.y = y

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.
        Parameters:
            params (dict): Dictionary of hyperparameter tuning parameters.
        Returns:
            dict: Dictionary containing the loss (negative Accuracy score) and status.
        """
        n_estimators = int(params['n_estimators'])

        # Define the model and the parameters
        clf = IsolationForest(n_estimators=n_estimators, random_state=42)
        # Fit the model
        clf.fit(self.X)
        # Predict the labels
        y_pred = clf.predict(self.X)  # Outlier labels (1 = outliers & -1 = inliners)
        y_pred = (y_pred == -1).astype(int)  # Convert the labels (-1, 1) to (0, 1)
        # Calculate the decision scores
        y_score = -clf.decision_function(self.X)  # Raw label scores

        # Calculate the performance scores
        roc_auc = round(roc_auc_score(self.y, y_score, average='weighted'), 3)

        return {'loss': -roc_auc, 'status': STATUS_OK, 'roc_auc_score': roc_auc}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.
        Returns:
            dict: Best number of neighbors obtained from tuning.
        """
        # Define the space to search for the optimized parameters
        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 10)
        }
        trials = Trials()
        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters:', best)

        return int(best['n_estimators'])

'''
================================
XGBoost parameter optimization
================================
'''
class XGBoost_tuner:
    """
    Hyperparameter tuning for XGBoost
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.
    Examples usage:
    xgboost_tuner=XGBoost_tuner(X, y)
    best_params = xgboost_tuner.tune_model() --> Used to train the final XGBoost model with optimal parameters.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.
        Parameters:
        params (dict): Dictionary of hyperparameter tuning parameters.
        Returns:
        dict: Dictionary containing the loss (negative Accuracy score) and status.
        """
        max_depth = int(params['max_depth'])
        n_estimators = int(params['n_estimators'])
        learning_rate = params['learning_rate']

        # Define the model and its parameters
        clf = XGBClassifier(max_depth = max_depth,
                            n_estimators = n_estimators,
                            learning_rate = learning_rate)
        # Calculate the performance metrics with cross validation for each fold
        roc_auc_xgboost = make_scorer(roc_auc_score, average='weighted')
        results = cross_val_score(clf, X=self.X, y=self.y, cv=self.kf, scoring=roc_auc_xgboost)
        # Calculate the mean accuracy score for all folds
        mean_roc_auc = results.mean()
        return {'loss': -mean_roc_auc, 'status': STATUS_OK, 'roc_auc_score': mean_roc_auc}

    def tune_model(self):
        '''
        Performs hyperparameter tuning using Bayesian optimization.
        Returns:
            dict: Dictionary containing the loss (negative Accuracy score) and status.
        '''

        # Define the space to search for the optimized parameters
        space = {
            'max_depth': hp.quniform('max_depth', 2, 10, 1),
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 10),
            'learning_rate': hp.loguniform('learning_rate', -5, 0)
        }

        trials = Trials()
        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters:', best)

        return best

'''
================================
CatBoost parameter optimization
================================
'''
class Catboost_tuner:
    """
    Hyperparameter tuning for CatBoost classifier.
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.
    Examples usage:
    catboost_tuner=Catboost_tuner(X, y)
    best_params = catboost_tuner.tune_model() --> Used to train the final CatBoost model with optimal parameters.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.
        Parameters:
            params (dict): Dictionary of hyperparameter tuning parameters.
        Returns:
            dict: Dictionary containing the loss (negative Accuracy score) and status.
        """
        iterations = int(params['iterations'])
        learning_rate = params['learning_rate']
        depth = int(params['depth'])
        # Define the model and its parameters
        clf = CatBoostClassifier(
            iterations= iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='Logloss',
            verbose=False
        )
        # Calculate the performance metrics with cross validation for each
        roc_auc_catboost = make_scorer(roc_auc_score, average='weighted')
        results = cross_val_score(clf, X=self.X, y=self.y, cv=self.kf, scoring=roc_auc_catboost)
        # Calculate the mean accuracy score for all folds
        mean_roc_auc = results.mean()
        return {'loss': -mean_roc_auc, 'status': STATUS_OK, 'roc_auc_score': mean_roc_auc}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.
        Returns:
            dict: Best hyperparameters obtained from tuning.
        """
        # Define the space to search for the optimized parameters
        space = {
            'iterations': hp.quniform('iterations', 100, 1000, 10),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'depth': hp.quniform('depth', 4, 10, 1),
        }
        trials = Trials()
        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters:', best)

        return best

'''
================================
LOF parameter optimization
================================
'''

class LOF_tuner:
    """
    Hyperparameter tuning for Local Outlier Factor (LOF) classifier.
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.
    Examples usage:
    lof_tuner=LOF_tuner(X, y)
    best_n_neighbors = lof_tuner.tune_model() --> Used to train the final LOF model with optimal parameters.
    """
    def __init__(self, X, y):
        """
        Initialize the hyperparameter tuning object.
        Parameters:
            X (array-like): Training data features.
            y (array-like): Ground truth labels.
        """
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.
        Parameters:
            params (dict): Dictionary of hyperparameter tuning parameters.
        Returns:
            dict: Dictionary containing the loss (negative Accuracy score) and status.
        """
        n_neighbors = int(params['n_neighbors'])
        # Define the model and the parameters
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', n_jobs=-1)
        # Fit the model
        clf.fit(self.X)
        # Predict the labels
        y_pred = clf.fit_predict(self.X)  # Outlier labels (1 = outliers & -1 = inliners)
        y_pred = (y_pred == -1).astype(int)  # Convert LOF labels (-1, 1) to (1, 0)
        # Get the decision function scores
        y_score = -clf.negative_outlier_factor_

        # Calculate the performance scores
        roc_auc = round(roc_auc_score(self.y, y_score, average='weighted'), 3)
        return {'loss': -roc_auc, 'status': STATUS_OK, 'roc_auc_score': roc_auc}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.
        Returns:
            dict: Best number of neighbors obtained from tuning.
        """
        # Define the space to search for the optimized parameters
        space = {
            'n_neighbors': hp.quniform('n_neighbors', 3, 100, 1)
        }

        trials = Trials()
        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters:', best)
        return int(best['n_neighbors'])

'''
================================================
Random Forest Classifier parameter optimization
===============================================
'''

class RandomForest_tuner:
    """
    Hyperparameter tuning for Random Forest classifier.
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
        Methods:
            -objective(params): Defines the optimization objective for hyperparameter tuning.
            -tune_model(): Performs hyperparameter tuning using Random Forest classifier.
        Examples usage:
        random_forest_tuner=RandomForest_tuner(X, y)
        best_n_estimator = random_forest_tuner.tune_model() --> Used to train the final LOF model with optimal parameters.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.
            Parameters:
                params (dict): Dictionary of hyperparameter tuning parameters.
            Returns:
                dict: Dictionary containing the loss (negative Accuracy score) and status.
            """
        n_estimators = int(params['n_estimators'])
        max_depth = int(params['max_depth'])
        # Define the model
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

        # Calculate the performance metrics with cross validation for each fold
        roc_auc_rf = make_scorer(roc_auc_score, average='weighted')
        results = cross_val_score(clf, X=self.X, y=self.y, cv=self.kf, scoring=roc_auc_rf)
        # Calculate the mean accuracy score for all folds.
        mean_roc_auc = results.mean()
        return {'loss': -mean_roc_auc, 'status': STATUS_OK, 'roc_auc_score': mean_roc_auc}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.
        Returns:
            dict: Best number of neighbors obtained from tuning.
        """
        # Define the space to search for the optimized parameters
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 1000, 10),
            'max_depth': hp.quniform('max_depth', 2, 50, 1)
        }

        trials = Trials()
        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters:', best)
        return best

'''
================================
KNN parameter optimization
================================
'''
class KNN_tuner:
    """
    Hyperparameter tuning for KNN.
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
        Methods:
            -objective(params): Defines the optimization objective for hyperparameter tuning.
            -tune_model(): Performs hyperparameter tuning using Random Forest classifier.
        Examples usage:
        random_forest_tuner=RandomForest_tuner(X, y)
        best_n_estimator = random_forest_tuner.tune_model() --> Used to train the final KNN model with optimal parameters.
    """
    def __init__(self,X, y):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.
            Parameters:
                params (dict): Dictionary of hyperparameter tuning parameters.
            Returns:
                dict: Dictionary containing the loss (negative Accuracy score) and status.
            """
        n_neighbors = int(params['n_neighbors'])
        # Define the model
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        # Calculate the performance metrics with cross validation for each fold
        roc_auc_knn = make_scorer(roc_auc_score, average='weighted')
        results = cross_val_score(clf, X=self.X, y=self.y, cv=self.kf, scoring=roc_auc_knn)
        # Calculate the mean accuracy score for all folds.
        roc_auc = results.mean()

        return {'loss': -roc_auc, 'status': STATUS_OK, 'roc_auc_score': roc_auc}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.
        Returns:
            dict: Best number of neighbors obtained from tuning.
        """
        # Define the space to search for the optimized parameters
        space = {
            'n_neighbors': hp.quniform('n_neighbors', 2, 100, 1)
        }

        trials = Trials()
        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)
        return best

'''
================================
DBSCAN parameter optimization
================================
'''

class DBSCAN_tuner:
    """
    Hyperparameter tuning for K=means classifier.
    Parameters:
        X (array-like): Features.
        y (array-like): Labels.
    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.
    Examples usage:
    dbscan_tuner=DBSCAN_tuner(X, y)
    best_eps = dbscan_tuner.tune_model() --> Used to run the final DBSCAN model with optimal parameters.
    """
    def __init__(self, X, y):
        """
        Initialize the hyperparameter tuning object.
        Parameters:
            X (array-like):  Data features.
            y (array-like): Ground truth labels.
        """
        self.X = X
        self.y = y

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.
        Parameters:
            params (dict): Dictionary of hyperparameter tuning parameters.
        Returns:
            dict: Dictionary containing the loss (negative Accuracy score) and status.
        """
        eps = params['eps']
        min_samples = int(params['min_samples'])
        # Define model and its parameters
        clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        # Fit the model
        y_pred = clf.fit_predict(self.X)  # Outlier labels (1 = outliers & -1 = inliners)
        # Convert the prediction labels (-1, 1) to (1, 0)
        y_pred = (y_pred == -1).astype(int)
        # Calculate the performance scores
        roc_auc = round(roc_auc_score(self.y, y_pred, average='weighted'), 3)

        return {'loss': -roc_auc, 'status': STATUS_OK, 'roc_auc_score': roc_auc}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.
        Returns:
            dict: Best number of neighbors obtained from tuning.
        """
        # Define the space to search for the optimized parameters
        space = {
            'eps': hp.uniform('eps', 0.2, 1),
            'min_samples': hp.quniform('min_samples', 2, 200, 1)
        }

        trials = Trials()
        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)
        return best