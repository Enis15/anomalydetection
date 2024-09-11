import pandas as pd

from sklearn.preprocessing import StandardScaler, normalize
from imblearn.over_sampling import SMOTE
from utils.unsupervised_learning import model_ecod, model_copod
from utils.supervised_learning import model_knn
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/creditcard.csv')

# Drop irrelavant features
df = df.drop(['Unnamed: 0', 'trans_date_trans_time', 'trans_num', 'unix_time', 'dob', 'first', 'last', 'merch_zipcode'], axis = 1)

# Encoding categorical features with numerical variables
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

# Determining the X and y values
X = df.drop('is_fraud', axis=1)
y = df['is_fraud'].values

scaler = StandardScaler()
X_scale = scaler.fit_transform(X) # Standardize the data
# smote = SMOTE(random_state=42)
# X_re, y_re = smote.fit_resample(X, y)
#
# scaler = StandardScaler()
# X = scaler.fit_transform(X) # Standardize the data

scorer = {'f1_score': make_scorer(f1_score, average = 'weighted'), 'roc_auc': make_scorer(roc_auc_score, average = 'weighted')} # Metrics for cross validation performance
kf = KFold(n_splits=5, shuffle=True, random_state=42) # Fold splits

roc_auc, f1_score, runtime = model_knn(X_scale, y, 100, scorer, kf)
