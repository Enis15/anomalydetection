import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn
from utils.supervised_learning import model_xgboost
from utils.supervised_learning import model_svm
from utils.supervised_learning import model_nb
from utils.supervised_learning import model_rf
from utils.supervised_learning import model_cb

# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/Fraud.csv')
print(df.shape())#return the shape of the dataset

#Feature engineering: Dropping the columns 'nameOrig' & 'nameDest'; Encoding values to the column 'CASH_OUT'
df = df.drop({'nameOrig','nameDest'},axis=1)
df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

#Determining the X and y values
X = df.drop('isFraud', axis=1)
y = df['isFraud'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Running the algorithms
results_knn = model_knn(X_train, X_test, y_train, y_test, 5)
#results_xgboost = model_xgboost(X_train, X_test, y_train, y_test, 4)
#results_svm = model_svm(X_train, X_test, y_train, y_test)
#results_nb = model_nb(X_train, X_test, y_train, y_test)
#results_rf = model_rf(X_train, X_test, y_train, y_test, 4)
#results_catboost = model_cb(X_train, X_test, y_train, y_test, 2)