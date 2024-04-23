import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn
from utils.supervised_learning import model_xgboost
from utils.supervised_learning import model_svm
from utils.supervised_learning import model_nb
from utils.supervised_learning import model_rf
from utils.supervised_learning import model_cb

#Load the dataset
df = pd.read_csv('../data/datasets/Labeled_DS/metaverse_transactions_dataset.csv')
print(df.shape)
print(df.dtypes)

#Dropping irrelevant columns for the anomaly detection
df = df.drop(['timestamp', 'sending_address', 'receiving_address'], axis=1)

#Labeling columns of type 'object'
columns_obj = ['transaction_type', 'location_region', 'purchase_pattern', 'age_group']
for i in columns_obj:
    label = LabelEncoder()
    df[i] = label.fit_transform(df[i])

#Relabeling column target column 'anomaly', where low risk:0, moderate & high risk =1
pd.set_option('future.no_silent_downcasting', True) #Ensure downcasting behavior is consistent with future versions of pandas
df['anomaly'] = df['anomaly'].replace({'low_risk': 0, 'moderate_risk': 1, 'high_risk': 1})

#Determining the X and y values
X = df.drop('anomaly', axis=1)
y = df['anomaly'].values

#Split the df into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Running the algorithms
results_knn = model_knn(X_train, X_test, y_train, y_test, 5)
#results_xgboost = model_xgboost(X_train, X_test, y_train, y_test, 4)
#results_svm = model_svm(X_train, X_test, y_train, y_test)
#results_nb = model_nb(X_train, X_test, y_train, y_test)
#results_rf = model_rf(X_train, X_test, y_train, y_test, 4)
#results_catboost = model_cb(X_train, X_test, y_train, y_test, 2)