from catboost import CatBoostClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from hpsklearn import HyperoptEstimator, random_forest_classifier, k_neighbors_classifier, xgboost_classification, isolation_forest
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import f1_score, silhouette_score, accuracy_score

#Parameter tuning using hpsklearn for the available algorithms(Knn, RandomForest, XGBoost & IsolationForest)
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
    #Get the model from the dictionary based on the provided model
    model = model_dict.get(model_name.lower())

    if model is None:
        raise ValueError(f'Invalid model name: {model_name}')

    estim = HyperoptEstimator(classifier=model, algo=tpe.suggest, max_evals=50, trial_timeout=220)

    #Try fitting the model
    try:
        estim.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")
        return None
    #Return the best model hyperparameters
    return estim.best_model()

class Catboost_tune:
    """
    Hyperparameter tuning for CATBoost classifier.

    Parameters:
        X_train (array-like): Training data features.
        X_test (array-like): Testing data features.
        y_train (array-like): Training data labels.
        y_test (array-like): Testing data labels.

    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.

    Examples usage:
    catboost_tuner=Catboost_tune(X_train, X_test, y_train, y_test)
    best_params = catboost_tuner.tune_model() --> Used to train the final CATBoost model with optimal parameters.
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        #Initialize the Catboost_tune instance.
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
            dict: Dictionary containing the loss (negative F1 score) and status.
        """
        model = CatBoostClassifier(
            iterations=int(params['iterations']),
            learning_rate=params['learning_rate'],
            depth=int(params['depth']),
            loss_function='Logloss',
            verbose=False
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        f1_scores = f1_score(self.y_test, y_pred, average='weighted')

        return {'loss': -f1_scores, 'status': STATUS_OK, 'f1_score': f1_scores}

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

class LOF_tune:
    """
    Hyperparameter tuning for Local Outlier Factor (LOF) classifier.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.

    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.

    Examples usage:
    lof_tuner=LOF_tune(X_train, y_train)
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
            dict: Dictionary containing the loss (negative F1 score) and status.
        """
        n_neighbors = int(params['n_neighbors'])

        clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='minkowski', n_jobs=-1)

        y_pred = clf.fit_predict(self.X)

        f1_scores = f1_score(self.y, y_pred, average='weighted')

        return {'loss': -f1_scores, 'status': STATUS_OK}

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
        print('Best parameters: ', best)

        return int(best['n_neighbors'])


class Kmeans_tune:
    """
    Hyperparameter tuning for K=means classifier.

    Parameters:
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.

    Methods:
        -objective(params): Defines the optimization objective for hyperparameter tuning.
        -tune_model(): Performs hyperparameter tuning using Bayesian optimization.

    Examples usage:
    kmeans_tuner=Kmeans_tune(X_train, y_train)
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
            dict: Dictionary containing the loss (negative silhouette score) and status.
        """
        n_neighbors = int(params['n_neighbors'])

        clf = KMeans(n_clusters=n_neighbors, random_state=42)
        clf.fit(self.X)

        y_pred = clf.predict(self.X)

        f1score = f1_score(self.y, y_pred, average='weighted')

        return {'loss': -f1score, 'status': STATUS_OK}

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



