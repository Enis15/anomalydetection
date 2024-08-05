import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

# Import the models
from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pyod.models.iforest import IForest


'''
================================
Isolation Forest parameter optimization
================================
'''

class IsolationForest_tuner:
    """
    Hyperparameter tuning for Local Outlier Factor (LOF) classifier.

    Parameters:
        X (array-like): Features.
        y (array-like): Labels.

    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.

    Examples usage:
    if_tuner=IsolationForest_tuner(X, y)
    best_n_neighbors = if_tuner.tune_model() --> Used to train the final Isolation Forest model with optimal parameters.
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
        n_estimators = int(params['n_estimators'])
        accuracy_scores = [] # To store the accuracy score for each fold

        # Perform cross validation manually

        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Define the model

            clf = IForest(n_estimators=n_estimators, random_state=42)

            # Fit the model
            clf.fit(X_train)

            # Predict on labels as (0, 1)
            y_pred = clf.predict(X_test)

           # Calculate the accuracy scores and appended to the list
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # Calculate the mean accuracy score for all folds.
        mean_accuracy = np.mean(accuracy_scores)

        return {'loss': -mean_accuracy, 'status': STATUS_OK, 'accuracy_score': mean_accuracy}

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
        results = cross_val_score(clf, X=self.X, y=self.y, cv=kf, scoring='accuracy')

        # Calculate the mean accuracy score for all folds
        accuracy = results.mean()

        return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy_score': accuracy}

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

        # Calculate the performance metrics with cross validation for each fold
        results = cross_val_score(clf, X=self.X, y=self.y, cv=kf, scoring='accuracy')

        # Calculate the mean accuracy score for all folds
        accuracy = results.mean()

        return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy_score': accuracy}

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
        accuracy_scores = [] # To store the accuracy score for each fold

        # Perform cross validation manually
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Define the model
            clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', n_jobs=-1)

            # Fit the model
            clf.fit(X_train)

            # Predict the labels
            y_pred = clf.fit_predict(X_test)

            # Convert LOF labels (-1, 1) to (1, 0)
            y_pred = (y_pred == -1).astype(int)

            # Calculate the accuracy scores and appended to the list
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # Calculate the mean accuracy score for all folds.
        mean_accuracy = np.mean(accuracy_scores)

        return {'loss': -mean_accuracy, 'status': STATUS_OK, 'accuracy_score': mean_accuracy}

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

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)

        results = cross_val_score(clf, X=self.X, y=self.y, cv=5, scoring='accuracy')

        accuracy = results.mean()

        return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy_score': accuracy}

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

        clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

        results = cross_val_score(clf, X=self.X, y=self.y, cv=5, scoring='accuracy')

        accuracy = results.mean()

        return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy_score': accuracy}

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
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

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
        accuracy_scores = []

        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')

            # Fit the model
            clf.fit(self.X_train)

            # Predict the labels
            y_pred = clf.labels_

            # Transform the labels (-1, 1) to (0, 1)
            y_pred = (y_pred == -1).astype(int)

            # Calculate the accuracy score for each fold
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

        # Calculate the mean accuracy score for all folds
        mean_accuracy = np.mean(accuracy_scores)

        return {'loss': -mean_accuracy, 'status': STATUS_OK, 'accuracy_score': mean_accuracy}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.

        Returns:
            dict: Best number of neighbors obtained from tuning.
        """
        # Define the space to search for the optimized parameters
        space = {
            'eps': hp.uniform('eps', 0.2, 0.9),
            'min_samples': hp.quniform('min_samples', 10, 100, 1)
        }

        trials = Trials()

        # Save the best model
        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)

        return best