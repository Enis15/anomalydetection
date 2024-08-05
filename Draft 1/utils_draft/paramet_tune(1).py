from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from hpsklearn import HyperoptEstimator, random_forest_classifier, k_neighbors_classifier, xgboost_classification, isolation_forest
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, random_state=123, shuffle=True)

# Parameter tuning using hpsklearn for the available algorithms(Knn, RandomForest, XGBoost & IsolationForest)
def paramet_tune(X_train, y_train, model_name='knn'):
    """
    Hyperparameter tuning for machine learning algorithms using hpsklearn library.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        model_name (str, optional): Name of the hyperparameter tuning algorithm (default is 'knn').

    Returns:
        object: Best-tuned model hyperparameters.
    """
    model_dict = {
        'random_forest': random_forest_classifier('my_rf'),
        'knn': k_neighbors_classifier('my_knn'),
        'xgboost': xgboost_classification('my_xgb'),
        'isolation_forest': isolation_forest('my_isol'),
    }

    # Get the model from the dictionary based on the provided model
    model = model_dict.get(model_name.lower())

    if model is None:
        raise ValueError(f'Invalid model name: {model_name}')

    estimator = HyperoptEstimator(classifier=model, algo=tpe.suggest, max_evals=50, trial_timeout=220)

    try:
        estimator.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        return None
    # Return the best model hyperparameters
    return estimator.best_model()

class Catboost_tuner:
    """
    Hyperparameter tuning for CATBoost classifier.

    Parameters:
        X (array-like): Features.
        y (array-like): Labels.


    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.

    Examples usage:
    catboost_tuner=Catboost_tuner(X_train, X_test, y_train, y_test)
    best_params = catboost_tuner.tune_model() --> Used to train the final CATBoost model with optimal parameters.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.kf = KFold(n_splits=5, random_state=123, shuffle=True)

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.

        Parameters:
            params (dict): Dictionary of hyperparameter tuning parameters.

        Returns:
            dict: Dictionary containing the loss (negative Accuracy score) and status.
        """
        model = CatBoostClassifier(
            iterations=int(params['iterations']),
            learning_rate=params['learning_rate'],
            depth=int(params['depth']),
            loss_function='Accuracy',
            verbose=False
        )

        results = cross_val_score(model, X=self.X, y=self.y, cv=kf, scoring='accuracy')

        accuracy = results.mean()

        return {'loss': -accuracy, 'status': STATUS_OK, 'accuracy_score': accuracy']}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.

        Returns:
            dict: Best hyperparameters obtained from tuning.
        """
        space = {
            'iterations': hp.quniform('iterations', 100, 500, 10),
            'learning_rate': hp.loguniform('learning_rate', -5, 0),
            'depth': hp.quniform('depth', 4, 10, 1),
        }
        trials = Trials()

        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters:', best)

        return best

class LOF_tuner:
    """
    Hyperparameter tuning for Local Outlier Factor (LOF) classifier.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.

    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.

    Examples usage:
    lof_tuner=LOF_tuner(X_train, y_train)
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

    def objective(self, params):
        """
        Defines the optimization objective for hyperparameter tuning.

        Parameters:
            params (dict): Dictionary of hyperparameter tuning parameters.

        Returns:
            dict: Dictionary containing the loss (negative Accuracy score) and status.
        """
        n_neighbors = int(params['n_neighbors'])

        clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', n_jobs=-1)

        y_pred = clf.fit_predict(self.X)

        # Convert LOF labels (-1, 1) to (1, 0)
        y_pred = (y_pred == -1).astype(int)

        accuracy_scores = accuracy_score(self.y, y_pred)

        return {'loss': -accuracy_scores, 'status': STATUS_OK, 'accuracy_score': accuracy_scores}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.

        Returns:
            dict: Best number of neighbors obtained from tuning.
        """

        space = {
            'n_neighbors': hp.quniform('n_neighbors', 3, 30, 1)
        }

        trials = Trials()

        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters:', best)

        return int(best['n_neighbors'])


class Kmeans_tuner:
    """
    Hyperparameter tuning for K=means classifier.

    Parameters:
        X (array-like): Data features.
        y (array-like): Data labels.

    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.

    Examples usage:
    kmeans_tuner=Kmeans_tuner(X, y)
    best_n_neighbors = kmeans_tuner.tune_model() --> Used to train the final K-means model with optimal parameters.
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
        n_neighbors = int(params['n_neighbors'])

        clf = KMeans(n_clusters=n_neighbors, random_state=42)
        clf.fit(self.X)

        y_pred = clf.predict(self.X)

        accuracy_scores = accuracy_score(self.y, y_pred)

        return {'loss': -accuracy_scores, 'status': STATUS_OK, 'accuracy_score': accuracy_scores}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.

        Returns:
            dict: Best number of neighbors obtained from tuning.
        """

        space = {
            'n_neighbors': hp.quniform('n_neighbors', 2, 30, 1)
        }

        trials = Trials()

        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)

        return int(best['n_neighbors'])

class RandomForest_tuner:
    """
    Hyperparameter tuning for Random Forest classifier.
    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Ground truth labels.
        X_test (array-like): Testing data features.
        y_test (array-like): Testing data labels.

        Methods:
            -objective(params): Defines the optimization objective for hyperparameter tuning.
            -tune_model(): Performs hyperparameter tuning using Random Forest classifier.
        Examples usage:
        random_forest_tuner=RandomForest_tuner(X_train, y_train)
        best_n_estimator = random_forest_tuner.tune_model() --> Used to train the final LOF model with optimal parameters.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

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
        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)

        accuracy_scores = accuracy_score(self.y_test, y_pred)

        return {'loss': -accuracy_scores, 'status': STATUS_OK, 'accuracy_score': accuracy_scores}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.

        Returns:
            dict: Best number of neighbors obtained from tuning.
        """

        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 1000, 10),
            'max_depth': hp.quniform('max_depth', 2, 30, 1)
        }

        trials = Trials()

        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)

        return best

class KNN_tuner:
    """
    Hyperparameter tuning for Random Forest classifier.
    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Ground truth labels.
        X_test (array-like): Testing data features.
        y_test (array-like): Testing data labels.

        Methods:
            -objective(params): Defines the optimization objective for hyperparameter tuning.
            -tune_model(): Performs hyperparameter tuning using Random Forest classifier.
        Examples usage:
        random_forest_tuner=RandomForest_tuner(X_train, y_train)
        best_n_estimator = random_forest_tuner.tune_model() --> Used to train the final LOF model with optimal parameters.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

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
        clf.fit(self.X_train, self.y_train)

        y_pred = clf.predict(self.X_test)

        accuracy_scores = accuracy_score(self.y_test, y_pred)

        return {'loss': -accuracy_scores, 'status': STATUS_OK, 'accuracy_score': accuracy_scores}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.

        Returns:
            dict: Best number of neighbors obtained from tuning.
        """

        space = {
            'n_neighbors': hp.quniform('n_neighbors', 2, 30, 1)
        }

        trials = Trials()

        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)

        return best

class DBSCAN_tuner:
    """
    Hyperparameter tuning for K=means classifier.

    Parameters:
        X (array-like): Data features.
        y (array-like): Data labels.

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

        clf = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        clf.fit(self.X)

        y_pred = clf.labels_

        accuracy_scores = accuracy_score(self.y, y_pred)

        return {'loss': -accuracy_scores, 'status': STATUS_OK, 'accuracy_score': accuracy_scores}

    def tune_model(self):
        """
        Performs hyperparameter tuning using Bayesian optimization.

        Returns:
            dict: Best number of neighbors obtained from tuning.
        """

        space = {
            'eps': hp.uniform('eps', 0.2, 0.9),
            'min_samples': hp.quniform('min_samples', 10, 100, 1)
        }

        trials = Trials()

        best = fmin(fn=self.objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print('Best parameters: ', best)

        return best