
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from mlp_classifier import MLPClassifier 
clf = MLPClassifier(hidden_layer_sizes=[10], random_state=42)
clf.fit(X_train, y_train)

print(f'Train accuracy: {clf.score(X_train, y_train):.4f}')
print(f'Test accuracy: {clf.score(X_test, y_test):.4f}')

#clf = MLPClassifier(hidden_layer_sizes=[10], learning_rate_init=0.001, max_iter=1000, activation='logistic', solver='sgd',
#                    batch_size=len(X_train), random_state=42, verbose=1, n_iter_no_change=100)
#clf.fit(X_train, y_train)
#print(clf.score(X_train, y_train))
#print(clf.score(X_test, y_test))

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import HistGradientBoostingClassifier
#clf = LogisticRegression()
#clf = Perceptron()
#clf = HistGradientBoostingClassifier()
# clf.fit(X_train, y_train)
# print(clf.score(X_train, y_train))
# print(clf.score(X_test, y_test))