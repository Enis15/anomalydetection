from utils.supervised_learning import model_knn, model_xgboost, model_svm, model_cb, model_nb, model_rf
from utils.unsupervised_learning import model_lof, model_iforest, model_ecod, model_pca, model_kmeans, model_copod
from utils.paramet_tune import paramet_tune, Catboost_tune, LOF_tune, Kmeans_tune
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from multiprocessing import freeze_support
import matplotlib.pyplot as plt



# Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/Fraud.csv')
print(df.shape())

#Feature engineering: Dropping the columns 'nameOrig' & 'nameDest'; Encoding values to the column 'CASH_OUT'
df = df.drop({'nameOrig','nameDest'},axis=1)
df['type'] = df['type'].map({'CASH_OUT': 5, 'PAYMENT': 4, 'CASH_IN': 3, 'TRANSFER': 2, 'DEBIT': 1})

#Determining the X and y values
X = df.drop('isFraud', axis=1)
y = df['isFraud'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train, X_test)
