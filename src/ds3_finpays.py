import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from models.supervised_learning import model_knn
from models.supervised_learning import model_xgboost
from models.supervised_learning import model_svm
from models.supervised_learning import model_nb
from models.supervised_learning import model_rf
from models.supervised_learning import model_cb

#Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/fin_paysys/bs140513_032310.csv')
print(df.shape)

#Replacing categorical values with dummy values
cat_features = df.select_dtypes(include=['object']).columns
for col in cat_features:
    df[col] = df[col].astype('category')

#Categorical features to numerical features
df[cat_features] = df[cat_features].astype('category').apply(lambda x: x.cat.codes)

#Determining the X and y values
X = df.drop('fraud', axis=1)
y = df['fraud'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Running the algorithms
results_knn = model_knn(X_train, X_test, y_train, y_test, 5)
#results_xgboost = model_xgboost(X_train, X_test, y_train, y_test, 4)
#results_svm = model_svm(X_train, X_test, y_train, y_test)
#results_nb = model_nb(X_train, X_test, y_train, y_test)
#results_rf = model_rf(X_train, X_test, y_train, y_test, 4)
#results_catboost = model_cb(X_train, X_test, y_train, y_test, 2)