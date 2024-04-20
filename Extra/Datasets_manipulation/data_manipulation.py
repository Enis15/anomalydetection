import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Fuction to perform general steps on data preprocessing
def data_preprocessing(df, target_column, test_size=0.3, random_state=None):

    #Drop the missing values and replace them with NaN
    df.dropna(inplace=True)

    #Split the target variable
    X = df[target_column]
    y = df.drop(columns=[target_column])

    #Encode for categorical variables
    for column in X.select_dtypes(include='object').columns:
        label_encoder = LabelEncoder()
        X[column] = label_encoder.fit_transform(X[column])

    #Scale numerical features
    scaler = StandardScaler()
    numeric_columns = X.select_dtypes(include=['float', 'int']).columns
    X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    #Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
