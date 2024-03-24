from pyod.models.ecod import ECOD
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print
from pyod.utils.example import visualize

X_train, X_test, y_train, y_test = \
        generate_data(n_train=200,
                      n_test=100,
                      n_features=2,
                      contamination=0.1,
                      random_state=42)

# train ECOD detector
clf_name = 'ECOD'
clf = ECOD()

# you could try parallel version as well.
    # clf = ECOD(n_jobs=2)
clf.fit(X_train)

    # get the prediction labels and outlier scores of the training data
y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
y_train_scores = clf.decision_scores_  # raw outlier scores

    # get the prediction on the test data
y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
y_test_scores = clf.decision_function(X_test)  # outlier scores

    # evaluate and print the results
print("\nOn Training Data:")
evaluate_print(clf_name, y_train, y_train_scores)
print("\nOn Test Data:")
evaluate_print(clf_name, y_test, y_test_scores)

    # visualize the results
visualize(clf_name, X_train, y_train, X_test, y_test, y_train_pred, y_test_pred, show_figure=True, save_figure=False)