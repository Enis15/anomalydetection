from hpsklearn import HyperoptEstimator, random_forest_classifier, k_neighbors_classifier, xgboost_classification, isolation_forest

def paramet_tune(X_train, y_train, model_name='knn'):

    model_dict = {
        'random_forest': random_forest_classifier('my_rf'),
        'knn': k_neighbors_classifier('my_knn'),
        'xgboost': xgboost_classification('my_xgb'),
        'isolation_forest': isolation_forest('my_isol'),
    }

    model = model_dict.get(model_name.lower())

    if model is None:
        raise ValueError(f'Invalid model name: {model_name}')

    estim = HyperoptEstimator(classifier=model)
    estim.fit(X_train, y_train)

    return estim.best_model()