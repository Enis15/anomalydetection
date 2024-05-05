from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score

#Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

#Determining the X and y values
X = df.drop('Class', axis=1)
y = df['Class'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'alpha': 10,
    'learning_rate': 0.05,
    'n_estimators': 100,
}

xgb_clf = XGBClassifier(**params)
xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)

f1_scores = f1_score(y_test, y_pred, average='weighted')
print(f1_scores)
roc_auc_scores = roc_auc_score(y_test, y_pred)
print(roc_auc_scores)
