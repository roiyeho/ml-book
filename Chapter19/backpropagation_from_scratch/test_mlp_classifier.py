# Author: Roi Yehoshua <roiyeho@gmail.com>
# August 2024
# License: MIT

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlp_classifier import MLPClassifier 

np.random.seed(42)

# Fetch the PIMA Indians Diabetes Dataset from OpenML
X, y = fetch_openml(name='diabetes', version=1, as_frame=False, return_X_y=True)

# Convert the targets into integers 
y = (y == 'tested_positive').astype(int)  # 0 = negative, 1 = positive

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train an MLP with default settings
clf = MLPClassifier(random_state=42)
clf.fit(X_train, y_train)

print(f'Train accuracy: {clf.score(X_train, y_train):.4f}')
print(f'Test accuracy: {clf.score(X_test, y_test):.4f}')

# Train an MLP with smaller size
clf2 = MLPClassifier(hidden_layer_sizes=[10], n_epochs=200, random_state=42)
clf2.fit(X_train, y_train)

print(f'Train accuracy: {clf2.score(X_train, y_train):.4f}')
print(f'Test accuracy: {clf2.score(X_test, y_test):.4f}')
