from sklearn.datasets import make_classification
from utils.unsupervised_learning import model_iforest, model_dbscan, model_copod, model_ecod, model_pca, model_lof
from sklearn.preprocessing import StandardScaler, normalize
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN
from sklearn.metrics import roc_auc_score, f1_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pyod.models.ecod import ECOD


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
#X, y = make_classification(n_samples=10000, n_features=15, n_classes=2, random_state=42)
#smote = SMOTE(random_state=42)
#X_re, y_re = smote.fit_resample(X, y)

scaler = StandardScaler()
X = scaler.fit_transform(X) # Standardize the data

X = normalize(X) # Normalize the data

# # Define model and its parameters
# clf = DBSCAN(eps=20, min_samples=150, metric='euclidean')
# # Fit the model
# clf.fit(X)
# # Get the prediction labels
# y_pred = clf.fit_predict(X)  # Outlier labels (1 = outliers & -1 = inliners)
# y_pred = (y_pred == -1).astype(int)  # Convert labels (-1, 1) to (1, 0)

# Define the model and the parameters
clf = ECOD()
# Fit the model
clf.fit(X)
# Predict the labels
y_pred = clf.predict(X)  # Labels (0, 1)
# Calculate the decision function scores
y_score = clf.decision_function(X)  # Raw label scores

# Calculate the performance scores
roc_auc_dbscan = round(roc_auc_score(y, y_pred, average='weighted'), 3)
f1_score_dbscan = round(f1_score(y, y_pred, average='weighted'), 3)
nmi_score_dbscan = round(normalized_mutual_info_score(y, y_pred), 3)
rand_score_dbscan = round(adjusted_rand_score(y, y_pred), 3)

# # Visualize the results
#
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X)
# plt.figure(figsize=(10, 7))
#
# plt.scatter(X_reduced[y_pred == 0, 0], X_reduced[y_pred == 0, 1],
#             c='blue', label='Inliers', s=20, alpha=0.6) # Plot inliers (predicted as 0)

# plt.scatter(X_reduced[y_pred == 1, 0], X_reduced[y_pred == 1, 1],
#             c='red', label='Outliers', s=30, alpha=0.6) # Plot outliers (predicted as 1)

# plt.title("DBSCAN Clustering Results")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend()
# plt.show()

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
matrix = ConfusionMatrixDisplay(cm, display_labels=['Inliners', 'Outliers'])

matrix.plot(cmap=plt.cm.Blues)
plt.show()