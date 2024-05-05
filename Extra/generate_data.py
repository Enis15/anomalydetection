from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.supervised_learning import model_knn
X, y = make_classification(n_samples=10000, n_classes=2 ,n_features=15, random_state=5)

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
knn = model_knn(X_train, X_test, y_train, y_test, 5)